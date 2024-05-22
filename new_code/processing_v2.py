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
from PIL import Image, ImageFilter
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage, Table, TableStyle, SimpleDocTemplate, Spacer, Paragraph, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from model.model import Point, GlareRequestModel

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
        delta_lon = (point['longitude'] - origin_lon) * 111319.9 * math.cos(math.radians(origin_lat))
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

# Commenting out the function to plot results for CPs
# def plot_results_for_op(cps, ops_transformed, pv_points_transformed, df, op_index):
#     cps_array = np.array(cps)
#     cp_counts = df['index_cp'].value_counts().to_dict()
#     colors = [cp_counts.get(i, 0) for i in range(len(cps))]

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     # Scatter plot for CPs (latitude on x-axis, longitude on y-axis)
#     scatter = ax.scatter(cps_array[:, 1], cps_array[:, 0], cps_array[:, 2], c=colors, cmap='viridis', label='CP')

#     # Scatter plot for OPs
#     for op in ops_transformed:
#         ax.scatter(op[1], op[0], op[2], color='red', marker='x', s=100, label='OP')

#     # Plot PV field outline
#     pv_points_array = np.array(pv_points_transformed + [pv_points_transformed[0]], dtype='float')
#     ax.plot(pv_points_array[:, 1], pv_points_array[:, 0], pv_points_array[:, 2], color='green', linestyle='-', linewidth=2, label='PV Field')

#     ax.set_xlabel('Latitude (X)')
#     ax.set_ylabel('Longitude (Y)')
#     ax.set_zlabel('Height (Z)')

#     # Equal scaling
#     max_range = np.array([cps_array[:, 1].max() - cps_array[:, 1].min(), 
#                           cps_array[:, 0].max() - cps_array[:, 0].min(), 
#                           cps_array[:, 2].max() - cps_array[:, 2].min()]).max() / 2.0

#     mid_x = (cps_array[:, 1].max() + cps_array[:, 1].min()) * 0.5
#     mid_y = (cps_array[:, 0].max() + cps_array[:, 0].min()) * 0.5
#     mid_z = (cps_array[:, 2].max() + cps_array[:, 2].min()) * 0.5

#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)

#     # Top-down perspective (rotate view to look down from above)
#     ax.view_init(elev=90., azim=-90)

#     ax.set_title(f'Result for OP {op_index + 1}')
#     plt.colorbar(scatter, ax=ax, label='Minutes of glare per year')

#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'results_for_op_{op_index + 1}.png')
#     # plt.show()  # Commented out to prevent displaying images during execution

def plot_empty_results_for_op(op_index):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.text(0.5, 0.5, 0.5, 'No glare data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(f'Result for OP {op_index + 1}')
    plt.savefig(f'results_for_op_{op_index + 1}.png')
    plt.close()

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

def plot_sun_position_reflections_date_time(df, utc, i,timestamp):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    aggregated_df = df.groupby('timestamp')['superpositioned_by_sun'].agg(lambda x: False if False in x.values else True).reset_index()
    
    colors = aggregated_df['superpositioned_by_sun'].map({True: 'gray', False: 'gold'}).values
    
    decimal_hours = aggregated_df['timestamp'].dt.hour + aggregated_df['timestamp'].dt.minute / 60.0

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(aggregated_df['timestamp'], decimal_hours, c=colors, alpha=0.75, s=10, marker='o')

    legend_labels = {'gray': 'Sun Overlapping', 'gold': 'Glare Occurrence'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
               for color, label in legend_labels.items()]
    ax.legend(handles=handles, loc='upper left')

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.invert_yaxis()

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Time (UTC+{utc})')
    ax.set_title(f'Glare Periods for OP {i+1}')

    def time_formatter(x, pos):
        hours = int(x)
        minutes = int((x - hours) * 60)
        return f'{hours:02d}:{minutes:02d}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(time_formatter))
    
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    print('utc')
    print(utc)
    plt.savefig('assets/'+str(timestamp)+f'/sun_position_reflections_op_{i+1}.png')
    plt.close()

def plot_empty_sun_position_reflections(op_index,timestamp):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, 'No glare data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(f'Glare Periods for OP {op_index + 1}')
    plt.savefig('assets/'+str(timestamp)+f'/sun_position_reflections_op_{op_index + 1}.png')
    plt.close()

def plot_daily_glare_summary(daily_summary_dfs, utc):
    for i, daily_summary in enumerate(daily_summary_dfs):
        daily_summary['date'] = pd.to_datetime(daily_summary['date'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.bar(daily_summary['date'], daily_summary['glare'], label='Glare Duration', color='gold')
        ax.bar(daily_summary['date'], daily_summary['superpositioned_by_sun'], bottom=daily_summary['glare'], label='Sun Overlapping', color='gray')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('Month')
        ax.set_ylabel('Glare Duration per Day in Minutes')
        ax.set_title(f'Glare Duration per Day for OP {i+1}')
        ax.legend()
        
        ax.grid(True)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('assets/'+str(utc)+f'/glare_summary_op_{i+1}.png')
        plt.close()

def plot_empty_daily_glare_summary(op_index,sim_id):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, 'No glare data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(f'Glare Duration per Day for OP {op_index + 1}')
    print(sim_id)
    plt.savefig(f'assets/{sim_id}/glare_summary_op_{op_index + 1}.png')
    plt.close()

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

def add_summary_table(daily_summary_dfs, simplified_dfs, num_ops):
    elements.append(Paragraph("Glare Summary for Observation Points", styles['Heading2']))
    elements.append(Spacer(1, 12))
    summary_data = [["OP", "Total Glare\nper Year", "Max Glare\nper Day", "Total Glare per Year minus\nSun Overlap", "Max Glare per Day minus\nSun Overlap"]]

    for i in range(num_ops):
        if i < len(daily_summary_dfs):
            daily_summary = daily_summary_dfs[i]
            simplified_df = simplified_dfs[i]

            total_glare_per_year_minus_overlap = simplified_df[~simplified_df['superpositioned_by_sun']].shape[0]
            max_glare_per_day_minus_overlap = simplified_df[~simplified_df['superpositioned_by_sun']].groupby(simplified_df['timestamp'].dt.date).size().max()

            total_glare_per_year = simplified_df.shape[0]
            max_glare_per_day = simplified_df.groupby(simplified_df['timestamp'].dt.date).size().max()
        else:
            total_glare_per_year_minus_overlap = 0
            max_glare_per_day_minus_overlap = 0
            total_glare_per_year = 0
            max_glare_per_day = 0

        summary_data.append([f"OP {i+1}", f"{total_glare_per_year} min", f"{max_glare_per_day} min", f"{total_glare_per_year_minus_overlap} min", f"{max_glare_per_day_minus_overlap} min"])

    table = Table(summary_data, colWidths=[60, 80, 80, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

def add_pv_area_info(pv_areas, list_of_pv_area_information):
    elements.append(Paragraph("PV Area Information", styles['Heading2']))
    elements.append(Paragraph("Position of Corner Points", styles['Heading3']))
    elements.append(Spacer(1, 12))
    pv_data = [["Name", "Latitude", "Longitude", "Ground Elevation", "Height Above\nGround"]]
    for area, info in zip(pv_areas, list_of_pv_area_information):
        for point in area[:-1]:  # Remove the last point
            pv_data.append([info['name'], point['latitude'], point['longitude'], f"{point['ground_elevation']} m", f"{point['height_above_ground']} m"])
    table = Table(pv_data, colWidths=[80, 80, 80, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))
    
    elements.append(Paragraph("PV Module Information", styles['Heading3']))
    elements.append(Spacer(1, 12))
    pv_module_data = [["Name", "Azimuth", "Tilt", "Lower Edge Height", "Upper Edge Height"]]
    for info in list_of_pv_area_information:
        lower_edge_height = info.get('ground_elevation', 0)
        upper_edge_height = lower_edge_height + info.get('height_above_ground', 0)
        pv_module_data.append([info['name'], f"{info['azimuth']}°", f"{info['tilt']}°", f"{lower_edge_height} m", f"{upper_edge_height} m"])
    table = Table(pv_module_data, colWidths=[80, 80, 80, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

def add_op_info(ops):
    elements.append(Paragraph("Observation Points Information", styles['Heading2']))
    elements.append(Spacer(1, 12))
    op_data = [["OP", "Latitude", "Longitude", "Ground Elevation", "Height Above\nGround"]]
    for i, op in enumerate(ops):
        op_data.append([f"OP {i+1}", op['latitude'], op['longitude'], f"{op['ground_elevation']} m", f"{op['height_above_ground']} m"])
    table = Table(op_data, colWidths=[60, 80, 80, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0984E3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

def plot_geometry(pv_points_transformed_list, ops_transformed):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    pv_colors = ['green', 'lightgreen', 'darkgreen', 'lime', 'olive']
    
    # Plot PV areas
    for i, pv_points_transformed in enumerate(pv_points_transformed_list):
        pv_points_array = np.array(pv_points_transformed)
        pv_polygon = np.vstack([pv_points_array, pv_points_array[0]])
        ax.plot(pv_polygon[:, 0], pv_polygon[:, 1], pv_polygon[:, 2], color=pv_colors[i % len(pv_colors)], linestyle='-', linewidth=2, label=f'PV Area {i+1}')
    
    # Plot observation points (OPs)
    ops_array = np.array(ops_transformed)
    ax.scatter(ops_array[:, 0], ops_array[:, 1], ops_array[:, 2], color='red', s=100, marker='x', label='Observation Points (OP)')
    
    ax.set_xlabel('X (Longitude)')
    ax.set_ylabel('Y (Latitude)')
    ax.set_zlabel('Z (Elevation)')
    ax.legend()
    
    ax.set_title('3D Visualization of PV Areas and Observation Points')
    
    # Set equal scale for all axes
    all_points = np.vstack(pv_points_transformed_list + [ops_transformed])
    max_range = np.ptp(all_points, axis=0).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig('3d_geometry.png')
    plt.close()  # Commented out to prevent displaying images during execution


async def create_pdf(report_type, pv_areas, list_of_pv_area_information, ops, google_api_key, zoom_level, daily_summary_dfs, simplified_dfs, utc, meta_data):
    global elements, styles
    basePath = 'assets'+'/'

    outputPath =basePath+ str(meta_data['timestamp'])+'/'
    print(outputPath)
    doc = SimpleDocTemplate(outputPath+f"{report_type}_report.pdf", pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    logo_path = basePath+"logo.png"  # Path to the logo file

    # Title
    elements.append(RLImage(logo_path, width=100, height=50))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("PV-GlareCheck.com Simulation Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Project Information
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

    # Map Image
    map_image_path = generate_static_map(pv_areas, ops, google_api_key, zoom_level)
    if map_image_path and os.path.exists(map_image_path):
        elements.append(RLImage(map_image_path, width=540, height=360))
        elements.append(Spacer(1, 24))
    
    # 3D Geometry Plot
    plot_geometry([geographic_to_cartesian(area, ops[0]) for area in pv_areas], geographic_to_cartesian(ops, ops[0]))
    elements.append(RLImage(basePath+'3d_geometry.png', width=540, height=360))
    elements.append(PageBreak())

    # Summary Table
    add_summary_table(daily_summary_dfs, simplified_dfs, len(ops))
    elements.append(PageBreak())

    # Results for each OP
    for i in range(len(ops)):
        if i != 0:  # Only add PageBreak before results if it's not the first OP
            elements.append(PageBreak())
        elements.append(Paragraph(f"Results for OP {i+1}", styles['Heading2']))
        elements.append(Spacer(1, 12))
        if i < len(daily_summary_dfs):
            elements.append(RLImage(outputPath+f"glare_summary_op_{i+1}.png", width=400, height=267))
        else:
            plot_empty_daily_glare_summary(i,meta_data['timestamp'])
            elements.append(RLImage(outputPath+f"glare_summary_op_{i+1}.png", width=400, height=267))
        elements.append(Spacer(1, 12))

        if report_type == 'full':
            if i < len(daily_summary_dfs):
                elements.append(RLImage(outputPath+f"sun_position_reflections_op_{i+1}.png", width=400, height=267))
            else:
                plot_empty_sun_position_reflections(i,meta_data['timestamp'])
                elements.append(RLImage(outputPath+f"sun_position_reflections_op_{i+1}.png", width=400, height=267))
        elif report_type == 'free':
            if i < len(daily_summary_dfs):
                img = Image.open(outputPath+f"sun_position_reflections_op_{i+1}.png")
            else:
                plot_empty_sun_position_reflections(i,meta_data['timestamp'])
                img = Image.open(outputPath+f"sun_position_reflections_op_{i+1}.png")
            blurred_img = img.filter(ImageFilter.GaussianBlur(15))
            blurred_img_path = outputPath+f"blurred_sun_position_reflections_op_{i+1}.png"
            blurred_img.save(blurred_img_path)
            elements.append(RLImage(blurred_img_path, width=400, height=267))
            elements.append(Paragraph("Not all Details are visible in the free Report", styles['Normal']))

        elements.append(Spacer(1, 12))

    # PV Area Information and OP Information
    elements.append(PageBreak())  # Page break before PV Area Information
    add_pv_area_info(pv_areas, list_of_pv_area_information)
    add_op_info(ops)

    print("doc built")
    doc.build(elements)



async def process_data(pv_areas, list_of_pv_area_information, list_of_ops, meta_data, simulation_parameter, google_api_key):
    utc = meta_data['utc']
    timestamp = meta_data['timestamp']
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

    for i in range(len(list_of_ops)):
        if i < len(result_dfs):
            df = result_dfs[i]
            if df.empty:
                plot_empty_sun_position_reflections(i,str(timestamp))
                plot_empty_results_for_op(i)
            else:
                plot_sun_position_reflections_date_time(df, (utc), i,timestamp)
                # Commenting out the call to plot_results_for_op
                # if i < len(cps_set) and i < len(pv_areas_tansformed):
                #     plot_results_for_op(cps_set[i], ops_transformed, pv_areas_tansformed[i], df, i)
                # else:
                #     plot_empty_results_for_op(i)
        else:
            plot_empty_sun_position_reflections(i,str(timestamp))
            plot_empty_results_for_op(i)

    simplified_dfs = clean_and_simplify_dfs(result_dfs)

    daily_summary_dfs = create_daily_summary_dfs(simplified_dfs)
    
    plot_daily_glare_summary(daily_summary_dfs, meta_data['timestamp'])

    await create_pdf('full', pv_areas, list_of_pv_area_information, list_of_ops, google_api_key, simulation_parameter['zoom_level'], daily_summary_dfs, simplified_dfs, utc, meta_data)
    await create_pdf('free', pv_areas, list_of_pv_area_information, list_of_ops, google_api_key, simulation_parameter['zoom_level'], daily_summary_dfs, simplified_dfs, utc, meta_data)

if __name__ == '__main__':
    pv_point_01 = {'latitude': 53.590556, 'longitude': 10.078993, 'ground_elevation': 300, 'height_above_ground': 15}
    pv_point_02 = {'latitude': 53.590567, 'longitude': 10.079046, 'ground_elevation': 300, 'height_above_ground': 15}
    pv_point_03 = {'latitude': 53.590532, 'longitude': 10.079144, 'ground_elevation': 300, 'height_above_ground': 10}
    pv_point_04 = {'latitude': 53.590488, 'longitude': 10.078950, 'ground_elevation': 300, 'height_above_ground': 10}
    op_01 = {'latitude': 53.590490, 'longitude': 10.079334, 'ground_elevation': 300, 'height_above_ground': 10}
    op_02 = {'latitude': 53.590500, 'longitude': 10.079300, 'ground_elevation': 300, 'height_above_ground': 10}

    current_datetime = datetime.datetime.now()
    timestamp = int(current_datetime.timestamp())

    pv_areas = [[pv_point_01, pv_point_02, pv_point_03, pv_point_04]]
    list_of_pv_area_information = [{'azimuth': 180, 'tilt': 20, 'name': 'TPV 1', 'ground_elevation': 300, 'height_above_ground': 15}]
    list_of_ops = [op_01, op_02]
    meta_data = {'user_id': 123456789, 'project_id': 123456789, 'sim_id': 123456789, 'timestamp': timestamp, 'utc': 1}
    simulation_parameter = {'grid_width': 0.7, 'resolution': '1min', 'sun_elevation_threshold': 4, 'beam_spread': 6.5, 'sun_angle': 0.5, 'sun_reflection_threshold': 10.5, 'zoom_level': 20}

    google_api_key = "AIzaSyBSwoYxoN9_6oma8thxLWTdIeQkTxsN5Rs"


    process_data(pv_areas, list_of_pv_area_information, list_of_ops, meta_data, simulation_parameter, google_api_key)
