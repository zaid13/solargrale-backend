import os
import math
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
import datetime
import pvlib
from pvlib.location import Location
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
import requests
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage, Table, TableStyle, SimpleDocTemplate, Spacer, Paragraph, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Register Lato fonts
def register_fonts():
    try:
        pdfmetrics.registerFont(TTFont('Lato', 'Lato-Regular.ttf'))
        pdfmetrics.registerFont(TTFont('Lato-Bold', 'Lato-Bold.ttf'))
    except Exception as e:
        print(f"Error registering fonts: {e}")
        raise

register_fonts()

def generate_static_map(pv_areas, ops, google_api_key, zoom_level):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"

    size = "800x800"
    maptype = "satellite"
    scale = "2"

    center_lat = sum(point['latitude'] for area in pv_areas for point in area) / sum(len(area) for area in pv_areas)
    center_lon = sum(point['longitude'] for area in pv_areas for point in area) / sum(len(area) for area in pv_areas)
    center = f"{center_lat},{center_lon}"

    paths = []
    for area in pv_areas:
        area.append(area[0])
        path_points = "|".join(f"{point['latitude']},{point['longitude']}" for point in area)
        paths.append(f"path=color:0x0000ff80|weight:2|fillcolor:0x0000ff20|{path_points}")

    markers = []
    for idx, op in enumerate(ops):
        markers.append(f"color:red|label:{idx+1}|{op['latitude']},{op['longitude']}")

    url = (f"{base_url}center={center}&zoom={zoom_level}&size={size}&scale={scale}&maptype={maptype}&"
           f"{'&'.join(paths)}&{'&'.join(f'markers={marker}' for marker in markers)}&key={google_api_key}")

    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save("map_image.png")
        return "map_image.png"
    else:
        print("Fehler bei der Anfrage:", response.status_code)
        return None

def geographic_to_cartesian(points, origin):
    cartesian_coordinates = []
    origin_lat = origin['latitude']
    origin_lon = origin['longitude']
    origin_elevation = origin['ground_elevation'] + origin['height_above_ground']
    for point in points:
        delta_lat = (point['latitude'] - origin_lat) * 111319.9
        delta_lon = (point['longitude'] - origin['longitude']) * 111319.9 * math.cos(math.radians(point['latitude']))
        delta_elevation = (point['ground_elevation'] + point['height_above_ground']) - origin_elevation
        cartesian_coordinates.append((delta_lon, delta_lat, delta_elevation))
    return cartesian_coordinates

def fit_plane_to_points(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    A, B, C, D = lstsq([x, y, [1]*len(points)], z)[0]

    return A, B, C, D

def generate_calculation_points(points, width):
    points = np.array(points)
    pca = PCA(n_components=2)
    points_transformed = pca.fit_transform(points)
    hull = ConvexHull(points_transformed)

    x_min, x_max = points_transformed[:, 0].min(), points_transformed[:, 0].max()
    y_min, y_max = points_transformed[:, 1].min(), points_transformed[:, 1].max()

    x_coords = np.arange(x_min, x_max + width, width)
    y_coords = np.arange(y_min, y_max + width, width)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    grid = np.array([point for point in grid if is_point_inside_hull(point, hull)])

    calculation_points_global = pca.inverse_transform(grid)
    return calculation_points_global.tolist()

def is_point_inside_hull(point, hull):
    deln = Delaunay(hull.points[hull.vertices])
    return deln.find_simplex(point) >= 0

def plot_results_for_op(cps, op_transformed, pv_points_transformed, df, op_index):
    cps_array = np.array(cps)
    cp_counts = df['index_cp'].value_counts().to_dict()
    colors = [cp_counts.get(i, 0) for i in range(len(cps))]
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(cps_array[:, 0], cps_array[:, 1], cps_array[:, 2], c=colors, cmap='viridis', label='CP')
    
    for op in op_transformed:
        ax.scatter(op[0], op[1], op[2], color='red', marker='x', label='OP')
    
    pv_points_array = np.array(pv_points_transformed + [pv_points_transformed[0]], dtype='float')
    ax.plot(pv_points_array[:, 0], pv_points_array[:, 1], pv_points_array[:, 2], color='green', linestyle='-', linewidth=2, label='PV Field')
    
    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')
    ax.set_zlabel('Height (Z)')
    
    ax.set_ylim(ax.get_ylim()[::-1])
    
    ax.set_title(f'Result for OP {op_index + 1}')
    plt.colorbar(scatter, ax=ax, label='Minutes of glare per year')
    
    plt.legend()
    plt.tight_layout()
    # plt.show()  # Commented out to prevent displaying images during execution

def calculate_sun_positions(origin_point, year, timezone_offset, delta_t='1min', sun_threshold=0):
    site_location = Location(latitude=origin_point['latitude'],
                             longitude=origin_point['longitude'],
                             tz=f"Etc/GMT{'+' if timezone_offset < 0 else '-'}{abs(timezone_offset)}",
                             altitude=origin_point['ground_elevation'] + origin_point['height_above_ground'])
    
    times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:59', freq=delta_t, tz=site_location.tz)
    
    solpos = site_location.get_solarposition(times)
    
    filtered_solpos = solpos[solpos['apparent_elevation'] > sun_threshold]
    
    result = filtered_solpos[['azimuth', 'apparent_elevation']].rename(columns={'apparent_elevation': 'sun_height'})
    
    return result

def calculate_3d_direction_vectors(op_transformed, cps):
    op_directions = []
    for op in op_transformed:
        directions = []
        for cp in cps:
            direction = np.subtract(cp, op)
            directions.append(direction)
        op_directions.append(directions)
    return op_directions

def calculate_reflected_vectors(op_directions, azimuth, tilt):
    azimuth_rad = np.radians(azimuth)
    tilt_rad = np.radians(tilt)
    normal = np.array([
        np.sin(tilt_rad) * np.sin(azimuth_rad),
        np.sin(tilt_rad) * np.cos(azimuth_rad),
        np.cos(tilt_rad)
    ])
    reflection_set_op = []
    for op in op_directions:
        reflected_vectors_op = []
        for v in op:
            v = np.array(v)
            v_reflected = v - 2 * np.dot(v, normal) * normal
            reflected_vectors_op.append(v_reflected)
        reflection_set_op.append(reflected_vectors_op)
    return reflection_set_op

def check_reflection_direction_efficient(reflected_vecs_list, sun_positions, vecs_op_to_cps_list, threshold, sun_reflection_angle_threshold, pv_area_information):
    azimuth_rad = np.radians(sun_positions['azimuth'].values)
    elevation_rad = np.radians(sun_positions['sun_height'].values)
    sun_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
    sun_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
    sun_z = np.sin(elevation_rad)
    sun_vectors = np.stack((sun_x, sun_y, sun_z), axis=-1)

    threshold_rad = np.radians(threshold)
    sun_angle_threshold_rad = np.radians(sun_reflection_angle_threshold)

    dataframes_list = []

    for pv_area_index, (reflected_vecs, vecs_op_to_cps) in enumerate(zip(reflected_vecs_list, vecs_op_to_cps_list)):
        pv_area_info = pv_area_information[pv_area_index]

        for op_index, (op_vec, op_to_cp_vec) in enumerate(zip(reflected_vecs, vecs_op_to_cps)):
            op_records = []
            op_vec_norm = np.linalg.norm(op_vec, axis=1, keepdims=True)
            op_vec_unit = op_vec / op_vec_norm
            op_to_cp_vec_norm = np.linalg.norm(op_to_cp_vec, axis=1, keepdims=True)
            op_to_cp_vec_unit = op_to_cp_vec / op_to_cp_vec_norm

            cos_angle = np.dot(op_vec_unit, sun_vectors.T)
            cos_angle_op_to_cp = np.dot(op_to_cp_vec_unit, sun_vectors.T)

            within_threshold = cos_angle > np.cos(threshold_rad)
            sun_superposition = cos_angle_op_to_cp > np.cos(sun_angle_threshold_rad)

            for sun_pos_index in range(len(sun_positions)):
                reflecting_indices = np.where(within_threshold[:, sun_pos_index])[0]
                for idx in reflecting_indices:
                    sun_pos = sun_positions.iloc[sun_pos_index]
                    op_records.append({
                        'timestamp': sun_positions.index[sun_pos_index],
                        'pv_area_index': pv_area_index,
                        'pv_area_name': pv_area_info['name'],
                        'index_cp': idx,
                        'sun_azimuth': sun_pos['azimuth'],
                        'sun_height': sun_pos['sun_height'],
                        'diff_angle': np.degrees(np.arccos(np.clip(cos_angle_op_to_cp[idx, sun_pos_index], -1.0, 1.0))),
                        'superpositioned_by_sun': sun_superposition[idx, sun_pos_index],
                        'azimuth': pv_area_info['azimuth'],
                        'tilt': pv_area_info['tilt'],
                    })
            
            if op_records:
                op_df = pd.DataFrame(op_records)
                dataframes_list.append(op_df)

    return dataframes_list

def plot_sun_position_reflections_date_time(df, utc, i):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    aggregated_df = df.groupby('timestamp')['superpositioned_by_sun'].agg(lambda x: False if False in x.values else True).reset_index()
    
    colors = aggregated_df['superpositioned_by_sun'].map({True: 'gray', False: 'gold'}).values
    
    decimal_hours = aggregated_df['timestamp'].dt.hour + aggregated_df['timestamp'].dt.minute / 60.0

    fig, ax = plt.subplots(figsize=(12, 6))

    sc = ax.scatter(aggregated_df['timestamp'], decimal_hours, c=colors, alpha=0.75, s=10, marker='o')

    legend_labels = {'gray': 'Sun overlapping reflection', 'gold': 'Glare occurrence'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
               for color, label in legend_labels.items()]
    ax.legend(handles=handles, loc='upper left')

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.invert_yaxis()

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Time (UTC+{utc})')
    ax.set_title(f'Detected glare periods for OP {i+1}')

    def time_formatter(x, pos):
        hours = int(x)
        minutes = int((x - hours) * 60)
        return f'{hours:02d}:{minutes:02d}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(time_formatter))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'sun_position_reflections_op_{i+1}.png')
    # plt.show()  # Commented out to prevent displaying images during execution

def plot_geometry(pv_points_transformed_list, cps_list, ops_transformed):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    pv_colors = ['green', 'lightgreen', 'darkgreen', 'lime', 'olive']
    cp_colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'white']
    
    for i, pv_points_transformed in enumerate(pv_points_transformed_list):
        pv_points_array = np.array(pv_points_transformed)
        pv_polygon = np.vstack([pv_points_array, pv_points_array[0]])
        ax.plot(pv_polygon[:, 0], pv_polygon[:, 1], pv_polygon[:, 2], color=pv_colors[i % len(pv_colors)], linestyle='-', linewidth=2, label=f'PV Polygon Group {i+1}')
    
    for i, cps in enumerate(cps_list):
        cps_array = np.array(cps)
        ax.scatter(cps_array[:, 0], cps_array[:, 1], cps_array[:, 2], color=cp_colors[i % len(cp_colors)], s=50, label=f'Calculation Points Group {i+1}')
    
    ops_array = np.array(ops_transformed)
    ax.scatter(ops_array[:, 0], ops_array[:, 1], ops_array[:, 2], color='red', s=100, marker='x', label='Observation Points (OP)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    ax.set_title('Visualization of PV Polygons, CPs and OPs')
    
    plt.tight_layout()
    # plt.show()  # Commented out to prevent displaying images during execution

def save_df_to_excel_with_timestamp(df):
    df_copy = df.copy()
    
    for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        df_copy[col] = df_copy[col].dt.tz_localize(None)

    current_time = datetime.datetime.now()
    
    timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    
    filename = f'data_{timestamp_str}.xlsx'
    
    df_copy.to_excel(filename, index=False)
    
    print(f'DataFrame was saved as: {filename}')

def clean_and_simplify_dfs(result_dfs):
    simplified_dfs = []
    for df in result_dfs:
        simplified_df = df[['timestamp', 'superpositioned_by_sun']].copy()
        
        simplified_df = simplified_df.groupby('timestamp')['superpositioned_by_sun'].agg(lambda x: False if False in x.values else True).reset_index()
        
        simplified_dfs.append(simplified_df)
    
    return simplified_dfs

def create_daily_summary_dfs(simplified_dfs):
    daily_summary_dfs = []
    for df in simplified_dfs:
        df['date'] = df['timestamp'].dt.date
        
        daily_summary = df.groupby('date').agg(
            glare=('superpositioned_by_sun', lambda x: (~x).sum()),
            superpositioned_by_sun=('superpositioned_by_sun', 'sum')
        ).reset_index()
        
        daily_summary_dfs.append(daily_summary)
        
    return daily_summary_dfs

def plot_daily_glare_summary(daily_summary_dfs, utc):
    for i, daily_summary in enumerate(daily_summary_dfs):
        daily_summary['date'] = pd.to_datetime(daily_summary['date'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(daily_summary['date'], daily_summary['glare'], label='Glare Duration', color='gold')
        ax.bar(daily_summary['date'], daily_summary['superpositioned_by_sun'], bottom=daily_summary['glare'], label='Sun Overlapping', color='gray')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Month')
        ax.set_ylabel('Glare Duration per Day in Minutes')
        ax.set_title(f'Glare Summary for OP {i+1}')

        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'glare_summary_op_{i+1}.png')
        # plt.show()  # Commented out to prevent displaying images during execution

def create_pdf(pv_areas, list_of_pv_area_information, ops, google_api_key, zoom_level, daily_summary_dfs, utc, meta_data):
    doc = SimpleDocTemplate("report.pdf", pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Use Lato font in the document
    styles.add(ParagraphStyle(name='Lato', fontName='Lato', fontSize=12))
    styles.add(ParagraphStyle(name='Lato-Bold', fontName='Lato-Bold', fontSize=12))
    styles.add(ParagraphStyle(name='Lato-Title', fontName='Lato-Bold', fontSize=16))
    styles.add(ParagraphStyle(name='Lato-Heading2', fontName='Lato-Bold', fontSize=14))
    styles.add(ParagraphStyle(name='Lato-Heading3', fontName='Lato-Bold', fontSize=12))
    
    width, height = A4

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Lato', 10)
        page_number_text = f"Page {doc.page}"
        canvas.drawString(A4[0] - 60, 40, page_number_text)
        canvas.restoreState()

    def add_project_info():
        project_info = [
            ["Project ID:", meta_data['project_id']],
            ["Simulation ID:", meta_data['sim_id']],
            ["Date:", datetime.datetime.fromtimestamp(meta_data['timestamp']).strftime("%Y-%m-%d")],
            ["App Version:", "1.0"]
        ]
        table = Table(project_info, colWidths=[100, 400])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Lato-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Lato'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

    def add_summary_table():
        elements.append(Paragraph("Glare Summary for Observation Points", styles['Lato-Heading2']))
        elements.append(Spacer(1, 10))

        summary_data = [["OP", "Max Glare per\nDay (min)", "Yearly Glare\n(min)", "Max Glare per Day\nminus Sun Overlap\n(min)", "Yearly Glare\nminus Sun Overlap\n(min)"]]
        for i, daily_summary in enumerate(daily_summary_dfs):
            max_glare_per_day = daily_summary['glare'].max()
            annual_total_glare = daily_summary['glare'].sum()
            annual_max_glare_minus_overlap = (daily_summary['glare'] - daily_summary['superpositioned_by_sun']).max()
            annual_total_glare_minus_overlap = (daily_summary['glare'] - daily_summary['superpositioned_by_sun']).sum()
            summary_data.append([f"OP {i+1}", max_glare_per_day, annual_total_glare, annual_max_glare_minus_overlap, annual_total_glare_minus_overlap])
        table = Table(summary_data, colWidths=[60, 80, 80, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Lato-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Lato'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

    def add_pv_area_info():
        elements.append(Paragraph("PV Area Information", styles['Lato-Heading2']))
        elements.append(Spacer(1, 10))
        pv_data = [["Name", "Latitude", "Longitude", "Ground Elevation\n(m)", "Height Above\nGround (m)"]]
        for area, info in zip(pv_areas, list_of_pv_area_information):
            for point in area[:-1]:  # Remove the last point
                pv_data.append([info['name'], point['latitude'], point['longitude'], f"{point['ground_elevation']} m", f"{point['height_above_ground']} m"])
        table = Table(pv_data, colWidths=[80, 80, 80, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Lato-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Lato'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

    def add_op_info():
        elements.append(Paragraph("Observation Points Information", styles['Lato-Heading2']))
        elements.append(Spacer(1, 10))

        op_data = [["OP", "Latitude", "Longitude", "Ground Elevation\n(m)", "Height Above\nGround (m)"]]
        for i, op in enumerate(ops):
            op_data.append([f"OP {i+1}", op['latitude'], op['longitude'], f"{op['ground_elevation']} m", f"{op['height_above_ground']} m"])
        table = Table(op_data, colWidths=[60, 80, 80, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Lato-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Lato'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

    # First page with title, project info, logo, and map
    logo_path = "logo.png"
    elements.append(RLImage(logo_path, width=100, height=50))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("PV-GlareCheck.com Simulation Report", styles['Lato-Title']))
    elements.append(Spacer(1, 12))
    add_project_info()
    map_image_path = generate_static_map(pv_areas, ops, google_api_key, zoom_level)
    if map_image_path:
        elements.append(RLImage(map_image_path, width=540, height=360))
        elements.append(Spacer(1, 24))
    elements.append(PageBreak())

    # Summary table on the second page
    add_summary_table()
    elements.append(PageBreak())

    # Results for each OP
    for i in range(len(daily_summary_dfs)):
        elements.append(Paragraph(f"Results for OP {i+1}", styles['Lato-Heading2']))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(f"glare_summary_op_{i+1}.png", width=400, height=267))
        elements.append(Spacer(1, 12))
        elements.append(RLImage(f"sun_position_reflections_op_{i+1}.png", width=400, height=267))
        elements.append(PageBreak())

    # PV Area Information and OP Information
    add_pv_area_info()
    add_op_info()

    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)

def process_data(pv_areas, list_of_pv_area_information, list_of_ops, meta_data, simulation_parameter, google_api_key):
    utc = meta_data['utc']
    date_from_timestamp = datetime.datetime.fromtimestamp(meta_data['timestamp'])
    year = date_from_timestamp.year
    resolution = simulation_parameter['resolution']
    sun_elevation_threshold = simulation_parameter['sun_elevation_threshold']
    threshold = (simulation_parameter['beam_spread'] + simulation_parameter['sun_angle']) / 2
    sun_reflection_angle_threshold = simulation_parameter['sun_reflection_threshold']
    grid_width = simulation_parameter['grid_width']

    origin_point = list_of_ops[0]

    pv_areas_tansformed = []
    for pv_area in pv_areas:
        pv_area_transformed = geographic_to_cartesian(pv_area, origin_point)
        pv_areas_tansformed.append(pv_area_transformed)
    
    ops_transformed = geographic_to_cartesian(list_of_ops, origin_point)

    cps_set = []
    for pv_area_transformed in pv_areas_tansformed:
        cps = generate_calculation_points(pv_area_transformed, grid_width)
        cps_set.append(cps)

    sun_positions = calculate_sun_positions(origin_point, year, utc, resolution, sun_elevation_threshold)
    
    plot_geometry(pv_areas_tansformed, cps_set, ops_transformed)

    set_of_vecs_op_to_cps = []
    for cps in cps_set:
        vecs_op_to_cps = calculate_3d_direction_vectors(ops_transformed, cps)
        set_of_vecs_op_to_cps.append(vecs_op_to_cps)
 
    set_of_reflected_vecs = []
    for i, vecs_op_to_cps in enumerate(set_of_vecs_op_to_cps):
        azimuth, tilt = list_of_pv_area_information[i]['azimuth'], list_of_pv_area_information[i]['tilt']
        reflected_vecs = calculate_reflected_vectors(vecs_op_to_cps, azimuth, tilt)
        set_of_reflected_vecs.append(reflected_vecs)
 
    result_dfs = check_reflection_direction_efficient(set_of_reflected_vecs, sun_positions, set_of_vecs_op_to_cps, threshold, sun_reflection_angle_threshold, list_of_pv_area_information)

    for i, df in enumerate(result_dfs):
        plot_sun_position_reflections_date_time(df, utc, i)

    simplified_dfs = clean_and_simplify_dfs(result_dfs)

    daily_summary_dfs = create_daily_summary_dfs(simplified_dfs)
    
    plot_daily_glare_summary(daily_summary_dfs, utc)

    create_pdf(pv_areas, list_of_pv_area_information, list_of_ops, google_api_key, simulation_parameter['zoom_level'], daily_summary_dfs, utc, meta_data)

if __name__ == '__main__':
    pv_point_01 = {'latitude': 53.590556, 'longitude': 10.078993, 'ground_elevation': 300, 'height_above_ground': 15}
    pv_point_02 = {'latitude': 53.590567, 'longitude': 10.079046, 'ground_elevation': 300, 'height_above_ground': 15}
    pv_point_03 = {'latitude': 53.590532, 'longitude': 10.079144, 'ground_elevation': 300, 'height_above_ground': 10}
    pv_point_04 = {'latitude': 53.590488, 'longitude': 10.078950, 'ground_elevation': 300, 'height_above_ground': 10}
    op_01 = {'latitude': 53.590490, 'longitude': 10.079334, 'ground_elevation': 300, 'height_above_ground': 10}
    op_02 = {'latitude': 53.590490, 'longitude': 10.079334, 'ground_elevation': 300, 'height_above_ground': 13}
    print('de')
    current_datetime = datetime.datetime.now()
    timestamp = int(current_datetime.timestamp())

    pv_areas = [[pv_point_01, pv_point_02, pv_point_03, pv_point_04]] 
    list_of_pv_area_information = [{'azimuth': 180, 'tilt': 20, 'name': 'TPV 1'}]
    list_of_ops = [op_02,op_01]
    meta_data = {'user_id': 123456789, 'project_id': 123456789, 'sim_id': 123456789, 'timestamp': timestamp, 'utc': 1}
    simulation_parameter = {'grid_width': 0.7, 'resolution': '1min', 'sun_elevation_threshold': 4, 'beam_spread': 6.5, 'sun_angle': 0.5, 'sun_reflection_threshold': 10.5, 'zoom_level': 20}

    google_api_key = "AIzaSyBSwoYxoN9_6oma8thxLWTdIeQkTxsN5Rs"

    process_data(pv_areas, list_of_pv_area_information, list_of_ops, meta_data, simulation_parameter, google_api_key)








