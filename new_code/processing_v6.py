import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, Point as ShapelyPoint
import pvlib
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import datetime
import math
import requests
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, Image as ReportLabImage, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image, ImageFilter
import time
import warnings
from reportlab.pdfgen.canvas import Canvas
from firebase_crud import uploadFileReturnUrl,addUrlTodocument,update_status


# Unterdrücke alle FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Funktion zur Umwandlung von beschreibenden Bezeichnern in numerische Werte
def convert_font_size(size):
    if isinstance(size, str):
        sizes = {
            'xx-small': 6,
            'x-small': 8,
            'small': 10,
            'medium': 12,
            'large': 14,
            'x-large': 16,
            'xx-large': 18,
            'smaller': 10,
            'larger': 14
        }
        return sizes.get(size, 12)
    return size

# Schriftgrößen um den Faktor 2 erhöhen, nur wenn sie numerisch sind
def scale_font_size(param):
    size = plt.rcParams[param]
    return convert_font_size(size) * 2

plt.rcParams.update({
    'font.size': scale_font_size('font.size'),
    'axes.titlesize': scale_font_size('axes.titlesize'),
    'axes.labelsize': scale_font_size('axes.labelsize'),
    'xtick.labelsize': scale_font_size('xtick.labelsize'),
    'ytick.labelsize': scale_font_size('ytick.labelsize'),
    'legend.fontsize': scale_font_size('legend.fontsize')
})

class MyPoint:
    def __init__(self, latitude, longitude, ground_elevation, height_above_ground):
        self.latitude = latitude
        self.longitude = longitude
        self.ground_elevation = ground_elevation
        self.height_above_ground = height_above_ground

    def __repr__(self):
        return (f"MyPoint(latitude={self.latitude}, longitude={self.longitude}, "
                f"ground_elevation={self.ground_elevation}, height_above_ground={self.height_above_ground})")

class PVArea:
    def __init__(self, points, azimuth, tilt, name):
        self.points = points
        self.azimuth = azimuth
        self.tilt = tilt
        self.name = name

    def __repr__(self):
        return (f"PVArea(name={self.name}, azimuth={self.azimuth}, tilt={self.tilt}, "
                f"points={self.points})")

class MetaData:
    def __init__(self, user_id, project_id, sim_id, timestamp, utc, project_name):
        self.user_id = user_id
        self.project_id = project_id
        self.sim_id = sim_id
        self.timestamp = timestamp
        self.utc = utc
        self.project_name = project_name

    def __repr__(self):
        return (f"MetaData(user_id={self.user_id}, project_id={self.project_id}, sim_id={self.sim_id}, "
                f"timestamp={self.timestamp}, utc={self.utc}, project_name={self.project_name})")

class SimulationParameter:
    def __init__(self, grid_width, resolution, sun_elevation_threshold, beam_spread, sun_angle, sun_reflection_threshold, zoom_level):
        self.grid_width = grid_width
        self.resolution = resolution
        self.sun_elevation_threshold = sun_elevation_threshold
        self.beam_spread = beam_spread
        self.sun_angle = sun_angle
        self.sun_reflection_threshold = sun_reflection_threshold
        self.zoom_level = zoom_level

    def __repr__(self):
        return (f"SimulationParameter(grid_width={self.grid_width}, resolution={self.resolution}, "
                f"sun_elevation_threshold={self.sun_elevation_threshold}, beam_spread={self.beam_spread}, "
                f"sun_angle={self.sun_angle}, sun_reflection_threshold={self.sun_reflection_threshold}, "
                f"zoom_level={self.zoom_level})")

def calculate_angles_and_distance(op, pv_areas):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def calculate_azimuth(lat1, lon1, lat2, lon2):
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        delta_lon = lon2 - lon1

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

        initial_azimuth = math.atan2(x, y)
        initial_azimuth = math.degrees(initial_azimuth)
        compass_azimuth = (initial_azimuth + 360) % 360

        return compass_azimuth

    data = []

    for op_index, op_point in enumerate(op, start=1):
        for area in pv_areas:
            for point in area.points:
                distance = haversine(op_point.latitude, op_point.longitude, point.latitude, point.longitude)
                elevation_diff = (point.ground_elevation + point.height_above_ground) - (op_point.ground_elevation + op_point.height_above_ground)
                elevation_angle = math.degrees(math.atan2(elevation_diff, distance))

                azimuth = calculate_azimuth(op_point.latitude, op_point.longitude, point.latitude, point.longitude)

                data.append([op_index, area.name, point.latitude, point.longitude, round(azimuth, 2), round(elevation_angle, 2)])

    return data

def calculate_reflection_direction(azimuth, elevation, panel_azimuth, panel_tilt):
    # Convert angles to radians
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    panel_azimuth_rad = math.radians(panel_azimuth)
    panel_tilt_rad = math.radians(panel_tilt)
    
    # Calculate the inverse sun vector (from sun to the point)
    sun_vector = np.array([
        -math.cos(elevation_rad) * math.sin(azimuth_rad),
        -math.cos(elevation_rad) * math.cos(azimuth_rad),
        -math.sin(elevation_rad)
    ])
    
    # Calculate the panel normal vector
    panel_normal = np.array([
        math.sin(panel_tilt_rad) * math.sin(panel_azimuth_rad),
        math.sin(panel_tilt_rad) * math.cos(panel_azimuth_rad),
        math.cos(panel_tilt_rad)
    ])
    
    # Calculate the reflection vector
    dot_product = np.dot(sun_vector, panel_normal)
    reflection_vector = sun_vector - 2 * dot_product * panel_normal
    
    # Convert the reflection vector to azimuth and elevation
    reflected_elevation = math.asin(reflection_vector[2])
    reflected_azimuth = math.atan2(reflection_vector[0], reflection_vector[1])
    
    reflected_azimuth = math.degrees(reflected_azimuth) % 360
    reflected_elevation = math.degrees(reflected_elevation)
    
    return round(reflected_azimuth, 2), round(reflected_elevation, 2)

def calculate_incidence_angle(azimuth, elevation, panel_azimuth, panel_tilt):
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    panel_azimuth_rad = math.radians(panel_azimuth)
    panel_tilt_rad = math.radians(panel_tilt)
    
    sun_vector = np.array([
        math.cos(elevation_rad) * math.sin(azimuth_rad),
        math.cos(elevation_rad) * math.cos(azimuth_rad),
        math.sin(elevation_rad)
    ])
    
    panel_normal = np.array([
        math.sin(panel_tilt_rad) * math.sin(panel_azimuth_rad),
        math.sin(panel_tilt_rad) * math.cos(panel_azimuth_rad),
        math.cos(panel_tilt_rad)
    ])
    
    incidence_angle_rad = math.acos(np.dot(sun_vector, panel_normal))
    incidence_angle = math.degrees(incidence_angle_rad)
    
    return round(incidence_angle, 2)

def generate_reflection_df(df_sun, pv_areas):
    # Prepare lists to store results
    pv_area_names = []
    timestamps = []
    sun_azimuths = []
    sun_elevations = []
    reflected_azimuths = []
    reflected_elevations = []
    inverse_azimuths = []
    inverse_elevations = []
    incidence_angles = []
    dnis = []

    # Convert columns to numpy arrays for faster computation
    sun_azimuth_arr = np.radians(df_sun['azimuth'].values)
    sun_elevation_arr = np.radians(df_sun['elevation'].values)
    dnis_arr = df_sun['dni'].values
    timestamps_arr = df_sun['timestamp'].values

    # Calculate the inverse sun vectors
    inv_sun_vectors = np.array([
        -np.cos(sun_elevation_arr) * np.sin(sun_azimuth_arr),
        -np.cos(sun_elevation_arr) * np.cos(sun_azimuth_arr),
        -np.sin(sun_elevation_arr)
    ]).T

    for pv_area in pv_areas:
        panel_azimuth_rad = np.radians(pv_area.azimuth)
        panel_tilt_rad = np.radians(pv_area.tilt)

        # Calculate the panel normal vector
        panel_normal = np.array([
            np.sin(panel_tilt_rad) * np.sin(panel_azimuth_rad),
            np.sin(panel_tilt_rad) * np.cos(panel_azimuth_rad),
            np.cos(panel_tilt_rad)
        ])

        # Calculate the dot product
        dot_products = np.dot(inv_sun_vectors, panel_normal)

        # Calculate reflection vectors
        reflection_vectors = inv_sun_vectors - 2 * dot_products[:, None] * panel_normal

        # Calculate reflected azimuth and elevation
        reflected_azimuths_arr = np.degrees(np.arctan2(reflection_vectors[:, 0], reflection_vectors[:, 1])) % 360
        reflected_elevations_arr = np.degrees(np.arcsin(reflection_vectors[:, 2]))

        # Calculate the incidence angles
        cos_incidence_angles = np.clip(dot_products, -1.0, 1.0)
        incidence_angles_arr = np.degrees(np.arccos(cos_incidence_angles))

        # Calculate inverse direction (adding 180 degrees to the azimuth)
        inverse_azimuths_arr = (reflected_azimuths_arr + 180) % 360
        inverse_elevations_arr = -reflected_elevations_arr

        # Append results to lists
        pv_area_names.extend([pv_area.name] * len(df_sun))
        timestamps.extend(timestamps_arr)
        sun_azimuths.extend(np.degrees(sun_azimuth_arr))
        sun_elevations.extend(np.degrees(sun_elevation_arr))
        reflected_azimuths.extend(reflected_azimuths_arr)
        reflected_elevations.extend(reflected_elevations_arr)
        inverse_azimuths.extend(inverse_azimuths_arr)
        inverse_elevations.extend(inverse_elevations_arr)
        incidence_angles.extend(incidence_angles_arr)
        dnis.extend(dnis_arr)

    # Create a DataFrame from the lists
    df_reflection = pd.DataFrame({
        'PV Area Name': pv_area_names,
        'timestamp': timestamps,
        'Sun Azimuth': sun_azimuths,
        'Sun Elevation': sun_elevations,
        'Reflected Azimuth': reflected_azimuths,
        'Reflected Elevation': reflected_elevations,
        'Inverse Azimuth': inverse_azimuths,
        'Inverse Elevation': inverse_elevations,
        'Incidence Angle': incidence_angles,
        'DNI (W/m²)': dnis
    })

    return df_reflection

def generate_points_within_angles(df, w):
    df_calculation_points = pd.DataFrame(columns=['OP Number', 'PV Area Name', 'Azimuth Angle', 'Elevation Angle'])

    for op_number in df['OP Number'].unique():
        for pv_area_name in df['PV Area Name'].unique():
            subset = df[(df['OP Number'] == op_number) & (df['PV Area Name'] == pv_area_name)]
            if subset.empty:
                continue

            # Normalize azimuth angles to be within [0, 360)
            polygon_points = subset[['Azimuth Angle', 'Elevation Angle']].values
            polygon_points[:, 0] = np.mod(polygon_points[:, 0], 360)
            
            # Create a new array to hold adjusted azimuths
            adjusted_polygon_points = []

            for i in range(len(polygon_points)):
                current_point = polygon_points[i]
                next_point = polygon_points[(i + 1) % len(polygon_points)]
                
                azimuth_diff = next_point[0] - current_point[0]
                
                # If the azimuth difference is greater than 180 degrees, we adjust the next point
                if azimuth_diff > 180:
                    next_point[0] -= 360
                elif azimuth_diff < -180:
                    next_point[0] += 360
                
                adjusted_polygon_points.append(current_point)
            
            adjusted_polygon_points = np.array(adjusted_polygon_points)
            
            # Create the polygon with adjusted azimuths
            polygon = Polygon([(x[0], x[1]) for x in adjusted_polygon_points])
            
            min_x, min_y, max_x, max_y = polygon.bounds
            x_range = np.arange(min_x, max_x + w, w)
            y_range = np.arange(min_y, max_y + w, w)
            
            points = []
            
            for x in x_range:
                for y in y_range:
                    point = ShapelyPoint(x, y)
                    if polygon.contains(point) or polygon.touches(point):
                        points.append((x % 360, y))
            
            new_data = pd.DataFrame(points, columns=['Azimuth Angle', 'Elevation Angle'])
            new_data['OP Number'] = op_number
            new_data['PV Area Name'] = pv_area_name
            
            if not new_data.empty:
                df_calculation_points = pd.concat([df_calculation_points, new_data], ignore_index=True)

    df_calculation_points = df_calculation_points.round(2)
    return df_calculation_points

def generate_sun_df(lat, lon, ground_elevation, timestamp, resolution='1min', sun_elevation_threshold=0):
    dt = datetime.datetime.fromtimestamp(timestamp)
    year = dt.year

    times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:59:59', freq=resolution, tz='UTC')

    location = pvlib.location.Location(lat, lon, 'UTC', ground_elevation)
    solpos = location.get_solarposition(times)
    dni = location.get_clearsky(times)['dni']

    df_sun = pd.DataFrame({
        'timestamp': times,
        'azimuth': solpos['azimuth'],
        'elevation': solpos['apparent_elevation'],
        'dni': dni
    })

    df_sun = df_sun[df_sun['elevation'] >= sun_elevation_threshold]
    df_sun.reset_index(drop=True, inplace=True)
    df_sun = df_sun.round(2)

    return df_sun

def generate_glare_results_efficient(df_reflection, df_calculation_points, beam_spread, sun_angle):
    glare_results_data = []
    df_calculation_points['number_of_hits'] = 0
    threshold = np.radians((beam_spread + sun_angle) / 2)

    # Convert relevant columns to numpy arrays for faster computation
    ref_azimuths = np.radians(df_reflection['Inverse Azimuth'].values)
    ref_elevations = np.radians(df_reflection['Inverse Elevation'].values)
    calc_azimuths = np.radians(df_calculation_points['Azimuth Angle'].values)
    calc_elevations = np.radians(df_calculation_points['Elevation Angle'].values)

    num_reflections = len(ref_azimuths)
    num_calc_points = len(calc_azimuths)

    # Preallocate result arrays
    hits_array = np.zeros((num_reflections, num_calc_points), dtype=bool)

    for i in range(num_reflections):
        ref_azimuth = ref_azimuths[i]
        ref_elevation = ref_elevations[i]

        # Calculate angle differences
        delta_azimuths = np.abs(calc_azimuths - ref_azimuth)
        delta_azimuths = np.minimum(delta_azimuths, 2 * np.pi - delta_azimuths)  # Handle wrapping around 360 degrees
        delta_elevations = np.abs(calc_elevations - ref_elevation)

        angle_diffs = np.sqrt(delta_azimuths**2 + delta_elevations**2)
        hits_array[i] = angle_diffs <= threshold

    hit_indices = np.where(hits_array)

    for hit_idx in range(len(hit_indices[0])):
        ref_idx = hit_indices[0][hit_idx]
        calc_idx = hit_indices[1][hit_idx]

        reflection_row = df_reflection.iloc[ref_idx]
        calc_row = df_calculation_points.iloc[calc_idx]

        glare_results_data.append([
            calc_row['OP Number'], reflection_row['PV Area Name'], reflection_row['timestamp'],
            reflection_row['DNI (W/m²)'], reflection_row['Incidence Angle'],
            reflection_row['Sun Azimuth'], reflection_row['Sun Elevation'],
            reflection_row['Reflected Azimuth'], reflection_row['Reflected Elevation'],
            reflection_row['Inverse Azimuth'], reflection_row['Inverse Elevation']
        ])
        df_calculation_points.at[calc_idx, 'number_of_hits'] += 1

    df_glare_results = pd.DataFrame(glare_results_data, columns=[
        'OP Number', 'PV Area Name', 'timestamp', 'DNI', 'Incidence Angle', 'Sun Azimuth',
        'Sun Elevation', 'Reflection Azimuth', 'Reflection Elevation', 'Inverse Azimuth', 'Inverse Elevation'
    ])

    df_glare_results = df_glare_results.round(2)
    return df_glare_results, df_calculation_points

def calculate_angle_difference_3d(azimuth1, elevation1, azimuth2, elevation2):
    azimuth1 = np.radians(azimuth1)
    elevation1 = np.radians(elevation1)
    azimuth2 = np.radians(azimuth2)
    elevation2 = np.radians(elevation2)
    
    x1 = np.cos(elevation1) * np.cos(azimuth1)
    y1 = np.cos(elevation1) * np.sin(azimuth1)
    z1 = np.sin(elevation1)
    
    x2 = np.cos(elevation2) * np.cos(azimuth2)
    y2 = np.cos(elevation2) * np.sin(azimuth2)
    z2 = np.sin(elevation2)
    
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    magnitude1 = np.sqrt(x1**2 + y1**2 + z1**2)
    magnitude2 = np.sqrt(x2**2 + y2**2 + z2**2)
    
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    return np.degrees(angle)

def calculate_direct_irradiance_on_plane(dni, sun_elevation, sun_azimuth, panel_tilt, panel_azimuth):
    dni = np.array(dni)
    sun_elevation = np.array(sun_elevation)
    sun_azimuth = np.array(sun_azimuth)
    panel_tilt = np.radians(panel_tilt)
    panel_azimuth = np.radians(panel_azimuth)

    sun_elevation_rad = np.radians(sun_elevation)
    sun_azimuth_rad = np.radians(sun_azimuth)

    cos_incidence = (
        np.sin(sun_elevation_rad) * np.cos(panel_tilt) +
        np.cos(sun_elevation_rad) * np.sin(panel_tilt) * np.cos(sun_azimuth_rad - panel_azimuth)
    )

    di_plane = dni * cos_incidence

    return di_plane

def add_di_plane_to_glare_results(df_glare_results, pv_areas):
    di_plane_list = []

    for index, row in df_glare_results.iterrows():
        pv_area = next(area for area in pv_areas if area.name == row['PV Area Name'])

        di_plane = calculate_direct_irradiance_on_plane(
            row['DNI'], row['Sun Elevation'], row['Sun Azimuth'], pv_area.tilt, pv_area.azimuth
        )

        di_plane_list.append(round(di_plane, 2))

    df_glare_results['di_plane'] = di_plane_list

    return df_glare_results

def load_module_data(filename, module_type=1):
    df = pd.read_csv(filename)
    df_filtered = df[df['ModuleType'] == module_type]
    x = df_filtered['Time'].values
    y = df_filtered['Value'].values
    return x, y

def fit_polynomial(x, y, degree=4):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    return poly

def add_luminance_to_glare_results(df_glare_results, poly_func, lf=125):
    luminance_list = []

    for index, row in df_glare_results.iterrows():
        incidence_angle = row['Incidence Angle']
        di_plane = row['di_plane']
        luminance = poly_func(incidence_angle) * (di_plane * lf) / 100000
        luminance_list.append(round(luminance, 2))

    df_glare_results['Luminance'] = luminance_list
    return df_glare_results

def aggregate_glare_results(df_glare_results):
    # Define a function to calculate the azimuth and elevation extents
    def calculate_extents(group):
        min_inverse_azimuth = group['Inverse Azimuth'].min()
        max_inverse_azimuth = group['Inverse Azimuth'].max()
        min_inverse_elevation = group['Inverse Elevation'].min()
        max_inverse_elevation = group['Inverse Elevation'].max()
        azimuth_extent = max_inverse_azimuth - min_inverse_azimuth
        elevation_extent = max_inverse_elevation - min_inverse_elevation
        return pd.Series({
            'DNI': group['DNI'].mean(),
            'Incidence Angle': group['Incidence Angle'].mean(),
            'Sun Azimuth': group['Sun Azimuth'].mean(),
            'Sun Elevation': group['Sun Elevation'].mean(),
            'Reflection Azimuth': group['Reflection Azimuth'].mean(),
            'Reflection Elevation': group['Reflection Elevation'].mean(),
            'Inverse Azimuth': group['Inverse Azimuth'].mean(),
            'Inverse Elevation': group['Inverse Elevation'].mean(),
            'di_plane': group['di_plane'].mean(),
            'Luminance': group['Luminance'].mean(),
            'Max Elevation Extent': elevation_extent,
            'Max Azimuth Extent': azimuth_extent
        })

    # Group by 'OP Number', 'PV Area Name', and 'timestamp' and aggregate
    df_aggregated = df_glare_results.groupby(['OP Number', 'PV Area Name', 'timestamp']).apply(calculate_extents).reset_index()
    return df_aggregated

def check_reflection_angle_threshold(df_glare_results, sun_reflection_threshold):
    def calculate_angle_difference_3d(azimuth1, elevation1, azimuth2, elevation2):
        azimuth1 = np.radians(azimuth1)
        elevation1 = np.radians(elevation1)
        azimuth2 = np.radians(azimuth2)
        elevation2 = np.radians(elevation2)
        
        x1 = np.cos(elevation1) * np.cos(azimuth1)
        y1 = np.cos(elevation1) * np.sin(azimuth1)
        z1 = np.sin(elevation1)
        
        x2 = np.cos(elevation2) * np.cos(azimuth2)
        y2 = np.cos(elevation2) * np.sin(azimuth2)
        z2 = np.sin(elevation2)
        
        dot_product = x1 * x2 + y1 * y2 + z1 * z2
        magnitude1 = np.sqrt(x1**2 + y1**2 + z1**2)
        magnitude2 = np.sqrt(x2**2 + y2**2 + z2**2)
        
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle)

    df_glare_results['Within_Threshold'] = df_glare_results.apply(
        lambda row: calculate_angle_difference_3d(row['Sun Azimuth'], row['Sun Elevation'], row['Inverse Azimuth'], row['Inverse Elevation']) <= sun_reflection_threshold,
        axis=1
    )

    return df_glare_results

def calculate_center_and_zoom(pv_areas, ops):
    # Collect all latitudes and longitudes
    latitudes = [point.latitude for area in pv_areas for point in area.points] + [op.latitude for op in ops]
    longitudes = [point.longitude for area in pv_areas for point in area.points] + [op.longitude for op in ops]

    # Calculate the center of the map
    center_lat = sum(latitudes) / len(latitudes)
    center_lng = sum(longitudes) / len(longitudes)

    # Calculate the zoom level
    max_lat_diff = max(latitudes) - min(latitudes)
    max_lng_diff = max(longitudes) - min(longitudes)
    max_diff = max(max_lat_diff, max_lng_diff)
    
    # Use the formula to calculate zoom level
    zoom = (math.floor(8 - math.log(max_diff) / math.log(2))) + 1
    if zoom > 20:
        zoom = 20

    return center_lat, center_lng, zoom

def generate_static_map_with_polygons(pv_areas, ops, api_key, output_dir, map_type="satellite", image_size="1280x1280"):
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    # Zentrum und Zoomstufe berechnen
    center_lat, center_lng, zoom = calculate_center_and_zoom(pv_areas, ops)

    # URL für die Google Static Maps API
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"

    # Marker und Beschriftungen für OPs hinzufügen
    markers = []
    for i, op in enumerate(ops, start=1):
        marker = f"color:red|label:{i}|{op.latitude},{op.longitude}"
        markers.append(marker)

    # Polygone für PV-Bereiche hinzufügen
    polygons = []
    for pv_area in pv_areas:
        path = "|".join(f"{point.latitude},{point.longitude}" for point in pv_area.points)
        # Sicherstellen, dass das Polygon geschlossen ist, indem der erste Punkt am Ende hinzugefügt wird
        first_point = f"{pv_area.points[0].latitude},{pv_area.points[0].longitude}"
        polygon = f"path=color:0x0000ff|weight:2|fillcolor:0x66b3ff33|{path}|{first_point}"
        polygons.append(polygon)

    # Alle Marker und Polygone zu einem einzelnen String kombinieren
    markers_str = "&".join([f"markers={marker}" for marker in markers])
    polygons_str = "&".join(polygons)

    # Den endgültigen URL zusammenstellen
    map_url = (f"{base_url}center={center_lat},{center_lng}&zoom={zoom}&size={image_size}"
               f"&maptype={map_type}&{markers_str}&{polygons_str}&key={api_key}")

    # Die Kartenabbildung anfordern
    print('map_url')
    print(map_url)
    response = requests.get(map_url)
    if response.status_code == 200:
        # Das Bild im Ausgabeverzeichnis speichern
        image_path = os.path.join(output_dir, 'pv_area_map.jpeg')  # Dateiendung auf '.jpeg' geändert
        with open(image_path, 'wb') as file:
            file.write(response.content)
        print(f"Map image saved to {image_path}")
    else:
        print(f"Failed to retrieve map image. Status code: {response.status_code}")

def save_image(image, image_path, dpi=300):
    for _ in range(3):  # Versuche es bis zu 3 Mal
        try:
            # Überprüfen Sie den Typ des Bildobjekts
            if isinstance(image, plt.Figure):  # Wenn es sich um ein Matplotlib-Figure-Objekt handelt
                image.savefig(image_path, dpi=dpi)
            elif isinstance(image, Image.Image):  # Wenn es sich um ein PIL-Image-Objekt handelt
                # Konvertieren Sie das Bild in den RGB-Modus, wenn es sich nicht bereits um einen RGB-Modus handelt
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, dpi=(dpi, dpi))
            
            if os.path.exists(image_path):
                # Überprüfen Sie, ob das Bild korrekt gespeichert wurde, indem Sie es öffnen
                with Image.open(image_path) as img:
                    img.verify()  # Dies wird eine Ausnahme auslösen, wenn das Bild beschädigt ist
                print(f"Bild erfolgreich gespeichert unter: {image_path}")
                return True
        except Exception as e:
            print(f"Fehler beim Speichern des Bildes {image_path}: {e}")
    print(f"Konnte das Bild nach 3 Versuchen nicht speichern: {image_path}")
    return False

def resize_image(image_path, max_width=5*inch, max_height=5*inch/ (16/10)):
    try:
        with Image.open(image_path) as img:
            aspect_ratio = img.width / img.height
            max_height = max_width / aspect_ratio
            img.thumbnail((max_width, max_height), Image.LANCZOS)
            resized_path = os.path.join(os.path.dirname(image_path), f"resized_{os.path.basename(image_path)}")
            if save_image(img, resized_path):
                return resized_path
            else:
                return None
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None


def blur_image(image_path, blur_radius=35):
    for _ in range(3):  # Try up to 3 times
        try:
            with Image.open(image_path) as img:
                img.load()  # Ensure the image is fully loaded
                img_blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
                blur_path = os.path.join(os.path.dirname(image_path), f"blur_{os.path.basename(image_path)}")
                img_blurred.save(blur_path)
                return blur_path
        except (OSError, IOError, SyntaxError) as e:
            print(f"Error processing image {image_path}: {e}")
            #time.sleep(0.1)  # Wait a bit before retrying
    return None


def generate_and_save_plots(df_pv_area_cornerpoints, df_calculation_points_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def adjust_azimuths(points):
        adjusted_points = []
        for i in range(len(points)):
            current_point = points[i]
            next_point = points[(i + 1) % len(points)]
            azimuth_diff = next_point[0] - current_point[0]
            if (next_point[0] - current_point[0]) > 180:
                next_point[0] -= 360
            elif (current_point[0] - next_point[0]) > 180:
                next_point[0] += 360
            adjusted_points.append(current_point)
        return np.array(adjusted_points)
    
    def transform_azimuth(azimuth):
        return (azimuth + 180) % 360 - 180
    
    for dp_number in df_pv_area_cornerpoints['OP Number'].unique():
        fig, ax = plt.subplots(figsize=(16, 9))
        dp_points_results = df_calculation_points_results[df_calculation_points_results['OP Number'] == dp_number]
        
        vmin = 0
        vmax = dp_points_results['number_of_hits'].max()
        if vmax <= vmin:
            vmax = vmin + 1

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
        cmap = plt.cm.inferno

        for pv_area_name in df_pv_area_cornerpoints['PV Area Name'].unique():
            pv_area_points_dp = df_pv_area_cornerpoints[(df_pv_area_cornerpoints['OP Number'] == dp_number) & (df_pv_area_cornerpoints['PV Area Name'] == pv_area_name)]
            if pv_area_points_dp.empty:
                continue
            
            azimuth_angles_dp = pv_area_points_dp['Azimuth Angle'].apply(transform_azimuth).tolist()
            elevation_angles_dp = pv_area_points_dp['Elevation Angle'].tolist()
            
            polygon_points = np.array(list(zip(azimuth_angles_dp, elevation_angles_dp)))
            adjusted_polygon_points = adjust_azimuths(polygon_points)
            
            azimuth_angles_dp = adjusted_polygon_points[:, 0].tolist() + [adjusted_polygon_points[0, 0]]
            elevation_angles_dp = adjusted_polygon_points[:, 1].tolist() + [adjusted_polygon_points[0, 1]]
            
            ax.plot(azimuth_angles_dp, elevation_angles_dp, color='black', linestyle='-', linewidth=3, zorder=1)
            
            center_x = np.mean(adjusted_polygon_points[:, 0])
            center_y = np.mean(adjusted_polygon_points[:, 1])
            ax.annotate(pv_area_name, xy=(center_x, center_y), xytext=(center_x + 30, center_y + 30),
                         arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, color='black', zorder=2)

        sc = ax.scatter(
            dp_points_results['Azimuth Angle'].apply(transform_azimuth),
            dp_points_results['Elevation Angle'],
            c=dp_points_results['number_of_hits'],
            cmap=cmap,
            norm=norm,
            s=10,
            zorder=3
        )
        plt.colorbar(sc, ax=ax, label='Minutes of Glare per Year')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_xlabel('Azimuth Angle (°) (0° = North, ±180° = South)')
        ax.set_ylabel('Elevation Angle (°)')
        ax.set_title(f'Perspective from DP {dp_number} onto the PV Areas - Including Glare Amount')
        ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
        
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, zorder=0)
        ax.axvline(x=90, color='k', linestyle='--', linewidth=0.5, zorder=0)
        ax.axvline(x=-90, color='k', linestyle='--', linewidth=0.5, zorder=0)
        ax.text(0, -100, 'North', horizontalalignment='center', verticalalignment='top', zorder=0)
        ax.text(90, -100, 'East', horizontalalignment='center', verticalalignment='top', zorder=0)
        ax.text(-90, -100, 'West', horizontalalignment='center', verticalalignment='top', zorder=0)
        ax.text(180, -100, 'South', horizontalalignment='center', verticalalignment='top', zorder=0)
        ax.text(-180, -100, 'South', horizontalalignment='center', verticalalignment='top', zorder=0)

        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

        fig.tight_layout()
        
        image_path = os.path.join(output_dir, f'reflecting_pv_area_dp_{dp_number}.png')
        if save_image(fig, image_path):
            blur_image(image_path)
        plt.close()

def plot_glare_data(df_aggregated, output_dir, timestamp, list_of_dps, utc_offset):
    os.makedirs(output_dir, exist_ok=True)
    
    year = pd.to_datetime(timestamp, unit='s').year
    print('keyss')
    print(df_aggregated.keys())
    pd.to_datetime(df_aggregated['timestamp'])
    df_aggregated['Date'] = pd.to_datetime(df_aggregated['timestamp']).dt.date

    df_aggregated['Time'] = pd.to_datetime(df_aggregated['timestamp']).dt.hour + pd.to_datetime(df_aggregated['timestamp']).dt.minute / 60.0

    utc_offset_str = f"UTC {utc_offset:+d}"

    for index, dp in enumerate(list_of_dps):
        op_number = index + 1
        df_op = df_aggregated[df_aggregated['OP Number'] == op_number]

        fig, ax = plt.subplots(figsize=(16, 10))
        if df_op.empty:
            ax.scatter([], [], color='yellow', label='Glare Occurrence')
            ax.scatter([], [], color='gray', label='Superimposed by Sun')
        else:
            for within_threshold, color, label in [(False, 'yellow', 'Glare Occurrence'), (True, 'gray', 'Superimposed by Sun')]:
                subset = df_op[df_op['Within_Threshold'] == within_threshold]
                ax.scatter(subset['Date'], subset['Time'], color=color, label=label, s=10)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
        ax.set_ylim(0, 24)
        ax.set_yticks(np.arange(0, 25, 1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):02d}:00'))
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Time of Day (HH:MM in {utc_offset_str})')
        ax.set_title(f'Glare Periods for DP {op_number}')
        
        # Ändere die Größe der Legendenmarkierungen
        legend = ax.legend()
        for handle in legend.legendHandles:
            handle._sizes = [30]  # Ändere die Größe der Legendenpunkte

        ax.grid(True, linestyle='--', linewidth=0.5)
        fig.tight_layout()
        image_path = os.path.join(output_dir, f'glare_periods_dp_{op_number}.png')
        if save_image(fig, image_path):
            blur_image(image_path)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 10))
        if df_op.empty:
            ax.set_ylim(0, 100)
            ax.set_xlabel('Date')
            ax.set_ylabel('Minutes per Day')
            ax.set_title(f'Glare Duration per Day for DP {op_number}')
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
            ax.grid(True, linestyle='--', linewidth=0.5)
        else:
            df_op['Day'] = pd.to_datetime(df_op['timestamp']).dt.date
            glare_duration = df_op.groupby(['Day', 'Within_Threshold']).size().unstack(fill_value=0)
            superimposed = glare_duration[True] if True in glare_duration.columns else pd.Series(0, index=glare_duration.index)
            glare_occurrence = glare_duration[False] if False in glare_duration.columns else pd.Series(0, index=glare_duration.index)
            ax.bar(glare_duration.index, glare_occurrence, label='Glare Occurrence', color='yellow')
            ax.bar(glare_duration.index, superimposed, bottom=glare_occurrence, label='Superimposed by Sun', color='gray')
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
            ax.set_ylim(0, max(glare_duration.sum(axis=1).max(), 100))
            ax.set_xlabel('Date')
            ax.set_ylabel('Minutes per Day')
            ax.legend()
            ax.grid(True, linestyle='--', linewidth=0.5)

        fig.tight_layout()
        ax.set_title(f'Glare Duration per Day for DP {op_number}')
        image_path = os.path.join(output_dir, f'glare_duration_dp_{op_number}.png')
        if save_image(fig, image_path):
            blur_image(image_path)
        plt.close()


def plot_glare_intensity_with_continuous_colorbar(df_aggregated, output_dir, timestamp, list_of_dps, utc_offset):
    os.makedirs(output_dir, exist_ok=True)

    year = pd.to_datetime(timestamp, unit='s').year
    
    df_aggregated['Date'] = pd.to_datetime(df_aggregated['timestamp']).dt.date
    df_aggregated['Time'] = pd.to_datetime(df_aggregated['timestamp']).dt.hour + pd.to_datetime(df_aggregated['timestamp']).dt.minute / 60.0

    df_max_luminance = df_aggregated.loc[df_aggregated.groupby(['OP Number', 'timestamp'])['Luminance'].idxmax()]

    utc_offset_str = f"UTC {utc_offset:+d}"

    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0, vmax=100000)

    for index, dp in enumerate(list_of_dps):
        op_number = index + 1
        df_op = df_max_luminance[df_max_luminance['OP Number'] == op_number]

        fig, ax = plt.subplots(figsize=(16, 10))
        sc = None
        if not df_op.empty:
            sc = ax.scatter(df_op['Date'], df_op['Time'], c=df_op['Luminance'], cmap=cmap, norm=norm, s=10)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
        ax.set_ylim(0, 24)
        ax.set_yticks(np.arange(0, 25, 1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):02d}:00'))
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Time of Day (HH:MM in {utc_offset_str})')
        ax.set_title(f'Glare Intensity for DP {op_number}')
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, label='Luminance (cd/m²)')
            cbar.set_ticks([0, 25000, 50000, 75000, 100000])
            cbar.set_ticklabels(['0', '25.000', '50.000', '75.000', '≥ 100.000'])
        ax.grid(True, linestyle='--', linewidth=0.5)
        fig.tight_layout()
        image_path = os.path.join(output_dir, f'glare_intensity_dp_{op_number}.png')
        if save_image(fig, image_path):
            blur_image(image_path)
        plt.close()

def save_aggregated_to_excel(df_aggregated, output_dir, file_name='aggregated_glare_results.xlsx'):
    """
    Save the aggregated glare results DataFrame to an Excel file.
    
    Parameters:
    df_aggregated (DataFrame): The DataFrame containing the aggregated glare results.
    output_dir (str): The directory where the Excel file will be saved.
    file_name (str): The name of the Excel file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert any datetime columns with timezone information to timezone-unaware
    for col in df_aggregated.columns:
        if pd.api.types.is_datetime64_any_dtype(df_aggregated[col]):
            if df_aggregated[col].dt.tz is not None:
                df_aggregated[col] = df_aggregated[col].dt.tz_localize(None)
    
    # Construct the full file path
    file_path = os.path.join(output_dir, file_name)
    
    # Save the DataFrame to an Excel file
    df_aggregated.to_excel(file_path, index=False)
    print(f'DataFrame saved to {file_path}')

def generate_summary(df_aggregated, list_of_dps):
    summary_data = []

    # Filter the data to include only entries where Within_Threshold is False
    df_filtered = df_aggregated[df_aggregated['Within_Threshold'] == False]

    # Group by OP Number and calculate the required metrics
    for op in list_of_dps:
        op_number = list_of_dps.index(op) + 1
        df_op = df_filtered[df_filtered['OP Number'] == op_number]

        if df_op.empty:
            max_glare_per_day = 0
            glare_per_year = 0
            days_with_relevant_glare = 0
        else:
            # Calculate glare per year
            glare_per_year = df_op.shape[0]

            # Calculate max. glare per day
            df_op.loc[:, 'Day'] = pd.to_datetime(df_op['timestamp']).dt.date
            glare_per_day = df_op.groupby('Day').size()
            max_glare_per_day = glare_per_day.max()

            # Calculate number of days with relevant glare
            days_with_relevant_glare = glare_per_day[glare_per_day > 0].count()

        summary_data.append({
            'OP Number': op_number,
            'Max. Glare per Day': max_glare_per_day,
            'Glare per Year': glare_per_year,
            'Days with Glare': days_with_relevant_glare
        })

    # Create a DataFrame from the summary data
    df_summary = pd.DataFrame(summary_data)
    
    return df_summary

# Function to add page numbers and header
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0] / 2.0, 0.5 * inch, text)
    header_text = f"PV-GlareCheck.com | Simulation Report | {meta_data.project_name}"
    canvas.drawCentredString(A4[0] / 2.0, A4[1] - 0.5 * inch, header_text)

class MyDocTemplate(SimpleDocTemplate):
    def __init__(self, filename, **kwargs):
        self.toc_entries = []
        super().__init__(filename, **kwargs)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in ["HeadingStyle"]:
                self.notify('TOCEntry', (1, flowable.getPlainText(), self.page))

class CustomCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        self.meta_data = kwargs.pop('meta_data', None)
        super().__init__(*args, **kwargs)

def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0] / 2.0, 0.5 * inch, text)
    header_text = f"PV-GlareCheck.com | Simulation Report | {canvas.meta_data.project_name}"
    canvas.drawCentredString(A4[0] / 2.0, A4[1] - 0.5 * inch, header_text)

def generate_reports(df_aggregated, output_dir, list_of_dps, meta_data, simulation_parameter, pv_areas):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, leading=28, spaceAfter=20))
    styles.add(ParagraphStyle(name='HeadingStyle', fontSize=18, leading=22, spaceAfter=15))
    styles.add(ParagraphStyle(name='SubHeadingStyle', fontSize=14, leading=18, spaceAfter=10))
    styles.add(ParagraphStyle(name='MetaStyle', fontSize=10, leading=12, spaceAfter=10))

    def create_document(report_type):
        doc_name = os.path.join(output_dir, f'{report_type}_report.pdf')
        doc = MyDocTemplate(doc_name, pagesize=A4, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        elements = []

        # Title page
        logo_path = os.path.join('assets', 'logo_v1.png')
        elements.append(Spacer(1, 1 * inch))

        with Image.open(logo_path) as img:
            width, height = img.size
            aspect_ratio = height / width
        max_width = 4 * inch
        img_width = max_width
        img_height = max_width * aspect_ratio

        elements.append(ReportLabImage(logo_path, width=img_width, height=img_height))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("Glare Simulation Report", styles['TitleStyle']))
        elements.append(Paragraph(meta_data.project_name, styles['TitleStyle']))
        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph(f"Full Report: All details are visible" if report_type == 'full' else "Free Report: Some details are not visible - please purchase full report on PV-GlareCheck.com in order to see all details", styles['SubHeadingStyle']))
        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph(f"User ID: {meta_data.user_id}", styles['MetaStyle']))
        elements.append(Paragraph(f"Project ID: {meta_data.project_id}", styles['MetaStyle']))
        elements.append(Paragraph(f"Simulation ID: {meta_data.sim_id}", styles['MetaStyle']))
        elements.append(Paragraph(f"Timestamp: {datetime.datetime.fromtimestamp(meta_data.timestamp)}", styles['MetaStyle']))
        elements.append(PageBreak())

        elements.append(Paragraph("1. Overview Map", styles['HeadingStyle']))
        elements.append(Paragraph("This map shows all PV areas and detection points.", styles['Normal']))
        
        map_image_path = os.path.join(output_dir, 'pv_area_map.jpeg')
        elements.append(ReportLabImage(map_image_path, 6 * inch, 6 * inch))
        elements.append(PageBreak())

        # Generiere die Zusammenfassung der Ergebnisse
        df_summary = generate_summary(df_aggregated, list_of_dps)

        # Erstelle die Tabelle mit angepassten Spaltenüberschriften und Einheiten
        summary_data = [
            ['DP Number', 'Max. Glare per Day [min]', 'Glare per Year [min]', 'Days with Glare']
        ]

        # Füge die Daten aus dem DataFrame hinzu
        for _, row in df_summary.iterrows():
            summary_data.append([
                row['OP Number'],
                row['Max. Glare per Day'],
                row['Glare per Year'],
                row['Days with Glare']
            ])

        # Erstelle die Tabelle mit den Spaltenbreiten
        summary_table = Table(summary_data, colWidths=[doc.width / len(summary_data[0])] * len(summary_data[0]))

        # Definiere den Stil der Tabelle
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        # Füge die Elemente zum PDF-Dokument hinzu
        elements.append(Paragraph("2. Summary of Results", styles['HeadingStyle']))
        elements.append(Paragraph("This table summarizes the results for each Detection Point (DP).", styles['Normal']))
        elements.append(summary_table)
        elements.append(PageBreak())

        elements.append(Paragraph("3. Simulation Parameters", styles['HeadingStyle']))
        sim_params_data = [
            ['Parameter', 'Value', 'Description'],
            ['Resolution', simulation_parameter.resolution, 'Time resolution for simulations.'],
            ['Sun Elevation\nThreshold', simulation_parameter.sun_elevation_threshold, 'Minimum sun elevation angle for calculating glare.'],
            ['Beam Spread', simulation_parameter.beam_spread, 'Spread of the reflected beam'],
            ['Sun Angle', simulation_parameter.sun_angle, 'Describes the visible sun size (diameter).'],
            ['Sun Reflection\nThreshold', simulation_parameter.sun_reflection_threshold, 'This parameter defines the limit for the difference angle\nbetween direct sunlight and reflection.\n If the angle is smaller/equal than the threshold, the sun is\nconsidered to superimpose the reflection (glare),\nmaking it irrelevant.']
        ]
        sim_params_table = Table(sim_params_data, colWidths=[doc.width * 0.2, doc.width * 0.2, doc.width * 0.6])
        sim_params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(Paragraph("This table shows the parameters used in the simulation.", styles['Normal']))
        elements.append(sim_params_table)
        elements.append(PageBreak())

        elements.append(Paragraph("4. PV Areas Details", styles['HeadingStyle']))
        pv_areas_data = [['PV Area', 'Azimuth [°]', 'Tilt [°]', 'Latitude', 'Longitude', 'Ground\nElevation\n[m]', 'Height\nAbove\nGround [m]']]
        for pv_area in pv_areas:
            for point in pv_area.points:
                pv_areas_data.append([pv_area.name, pv_area.azimuth, pv_area.tilt,
                                      point.latitude, point.longitude, point.ground_elevation, point.height_above_ground])
        pv_areas_table = Table(pv_areas_data, colWidths=[doc.width / len(pv_areas_data[0])] * len(pv_areas_data[0]))
        pv_areas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(Paragraph("This table lists the corner points of the PV areas.", styles['Normal']))
        elements.append(pv_areas_table)
        elements.append(PageBreak())

        elements.append(Paragraph("5. Detection Points (DPs) Details", styles['HeadingStyle']))
        dps_data = [['DP Number', 'Latitude', 'Longitude', 'Ground\nElevation\n[m]', 'Height\nAbove\nGround[m]']]
        for idx, dp in enumerate(list_of_dps, start=1):
            dps_data.append([idx, dp.latitude, dp.longitude, dp.ground_elevation, dp.height_above_ground])
        dps_table = Table(dps_data, colWidths=[doc.width / len(dps_data[0])] * len(dps_data[0]))
        dps_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(Paragraph("This table provides details of the detection points (DPs).", styles['Normal']))
        elements.append(dps_table)
        elements.append(PageBreak())

        width = 6 * inch
        height = 6 * inch
        aspect_ratio_n = 16 / 10

        for dp_number in range(1, len(list_of_dps) + 1):
            elements.append(Paragraph(f"6.{dp_number} Detection Point (DP) {dp_number}", styles['HeadingStyle']))
            if report_type == 'full':
                for i, plot_type in enumerate(['glare_periods', 'glare_duration', 'glare_intensity', 'reflecting_pv_area'], start=1):
                    image_path = os.path.join(output_dir, f'{plot_type}_dp_{dp_number}.png')
                    elements.append(Paragraph(f"This figure shows the {plot_type.replace('_', ' ')} for DP {dp_number}.", styles['Normal']))
                    elements.append(ReportLabImage(image_path, width, height / aspect_ratio_n))
                    if (i % 2) == 0:
                        elements.append(PageBreak())
            else:
                for i, plot_type in enumerate(['blur_glare_periods', 'glare_duration', 'blur_glare_intensity', 'blur_reflecting_pv_area'], start=1):
                    image_path = os.path.join(output_dir, f'{plot_type}_dp_{dp_number}.png')
                    elements.append(Paragraph(f"This figure shows the {plot_type.replace('_', ' ').replace('blur ', '')} for DP {dp_number}.", styles['Normal']))
                    elements.append(ReportLabImage(image_path, width, height / aspect_ratio_n))
                    if (i % 2) == 0:
                        elements.append(PageBreak())

        elements.append(Paragraph("7. Liability Disclaimer", styles['HeadingStyle']))
        elements.append(Paragraph(
            "The results presented in this report are based on simulations and calculations. "
            "While we strive to ensure the accuracy and reliability of the results, they may differ from real-world conditions. "
            "No liability is assumed for any damages or losses resulting from the use of these results. "
            "It is recommended to conduct further on-site evaluations and consider additional factors that may affect glare impact.",
            styles['Normal']))

        doc.build(elements, onFirstPage=lambda c, d: add_page_number(c, d), onLaterPages=lambda c, d: add_page_number(c, d), canvasmaker=lambda *args, **kwargs: CustomCanvas(*args, meta_data=meta_data, **kwargs))

    create_document('free')
    create_document('full')


"""
# Initialize Points
pv_point_01 = MyPoint(53.590556, 10.078993, 300, 15)
pv_point_02 = MyPoint(53.590567, 10.079046, 300, 15)
pv_point_03 = MyPoint(53.590532, 10.079144, 300, 10)
pv_point_04 = MyPoint(53.590488, 10.078950, 300, 10)
pv_point_05 = MyPoint(53.590500, 10.078732, 300, 15)
pv_point_06 = MyPoint(53.590521, 10.078823, 300, 15)
pv_point_07 = MyPoint(53.590475, 10.078898, 300, 10)
pv_point_08 = MyPoint(53.590440, 10.078711, 300, 10)
dp_03 = MyPoint(53.590490, 10.079334, 300, 10)
dp_02 = MyPoint(53.590635, 10.079033, 300, 10)
dp_01 = MyPoint(53.5903633, 10.079138, 300, 20)

current_datetime = datetime.datetime.now()
timestamp = int(current_datetime.timestamp())

pv_areas = [
    PVArea([pv_point_01, pv_point_02, pv_point_03, pv_point_04], 180, 20, 'TPV 1'),
    PVArea([pv_point_05, pv_point_06, pv_point_07, pv_point_08], 180, 20, 'TPV 2')
]

list_of_dps = [dp_01, dp_02, dp_03]

meta_data = MetaData(123456789, 123456789, 123456789, timestamp, 1, 'Projekt 1')

simulation_parameter = SimulationParameter(2, '1min', 4, 5, 0.5, 10, 20)
"""

def calculate_glare(_pv_areas, list_of_pv_area_information, list_of_ops, _meta_data, _simulation_parameter,api_key, output_dir,excluded_areas=[],):
    # Output path for images
    # output_dir = os.getcwd()
    print('api_key')
    print(api_key)
    # key_path = "api_key.txt"
    # # Öffne die Datei im Lesemodus ('r' für 'read')
    # with open(key_path, 'r') as datei:
    #     # Lese den Inhalt der Datei
    #     api_key = datei.read()
    plt.switch_backend('Agg')
    # Load module data and fit polynomial
    filename = 'assets/module_reflection_profiles.csv'
    module_type = 1
    x, y = load_module_data(filename, module_type)
    poly_func = fit_polynomial(x, y, degree=4)
    
    meta_data = MetaData(_meta_data['user_id'], _meta_data['project_id'], _meta_data['sim_id'], _meta_data['timestamp'], _meta_data['utc'], _meta_data['project_name'])
    simulation_parameter = SimulationParameter(_simulation_parameter['grid_width'], _simulation_parameter['resolution'], _simulation_parameter['sun_elevation_threshold'], _simulation_parameter['beam_spread'], _simulation_parameter['sun_angle'], _simulation_parameter['sun_reflection_threshold'], _simulation_parameter['zoom_level'])
    
    pv_areas = []
    for i, pvs in enumerate(_pv_areas):
        points = []
        for point in pvs:
            points.append(MyPoint(point['latitude'], point['longitude'], point['ground_elevation'], point['height_above_ground']))
        pv_areas.append(PVArea(points, list_of_pv_area_information[i]['azimuth'], list_of_pv_area_information[i]['tilt'], list_of_pv_area_information[i]['name']))

    list_of_dps = []
    for op in list_of_ops:
        list_of_dps.append(MyPoint(op['latitude'], op['longitude'], op['ground_elevation'], op['height_above_ground']))

    print("Schritt 1: Leite Positionen und Abstände der PV-Area-Eckpunkte ab.")
    data = calculate_angles_and_distance(list_of_dps, pv_areas)

    print("2")
    df_pv_area_cornerpoints = pd.DataFrame(data, columns=['OP Number', 'PV Area Name', 'Latitude', 'Longitude', 'Azimuth Angle', 'Elevation Angle'])

    print("3")
    df_calculation_points = generate_points_within_angles(df_pv_area_cornerpoints, w=simulation_parameter.grid_width)

    print("4")
    df_sun = generate_sun_df(list_of_dps[0].latitude, list_of_dps[0].longitude, list_of_dps[0].ground_elevation, meta_data.timestamp, simulation_parameter.resolution, simulation_parameter.sun_elevation_threshold)
    update_status(0.23,_meta_data['sim_id'])
    print("5")
    df_reflection = generate_reflection_df(df_sun, pv_areas)
    update_status(0.29,_meta_data['sim_id'])
    print("6")
    df_glare_results, df_calculation_points_results = generate_glare_results_efficient(df_reflection, df_calculation_points, simulation_parameter.beam_spread, simulation_parameter.sun_angle)
    update_status(0.34,_meta_data['sim_id'])

    print("7")
    df_glare_results = add_di_plane_to_glare_results(df_glare_results, pv_areas)
    update_status(0.36,_meta_data['sim_id'])

    print("8")
    df_glare_results = add_luminance_to_glare_results(df_glare_results, poly_func, lf=125)
    update_status(0.38,_meta_data['sim_id'])

    print("9")
    df_aggregated = aggregate_glare_results(df_glare_results)

    print("10")
    df_aggregated = check_reflection_angle_threshold(df_aggregated, simulation_parameter.sun_reflection_threshold)
    update_status(0.45,_meta_data['sim_id'])

    print("11")
    generate_static_map_with_polygons(pv_areas, list_of_dps, api_key, output_dir)
    update_status(0.52,_meta_data['sim_id'])

    print("12")
    generate_and_save_plots(df_pv_area_cornerpoints, df_calculation_points_results, output_dir)
    print("13")
    plot_glare_data(df_aggregated, output_dir, meta_data.timestamp, list_of_dps, meta_data.utc)
    update_status(0.58,_meta_data['sim_id'])

    print("14")
    plot_glare_intensity_with_continuous_colorbar(df_aggregated, output_dir, meta_data.timestamp, list_of_dps, meta_data.utc)
    print("15")
    # save_aggregated_to_excel(df_aggregated, output_dir)
    print("16")
    update_status(0.61,_meta_data['sim_id'])

    generate_reports(df_aggregated, output_dir, list_of_dps, meta_data, simulation_parameter, pv_areas)



def test():
    example_call = """
    {
      "identifier": "cub175",
      "pv_areas": [
        [
      {"latitude": 48.088565, "longitude": 11.566283, "ground_elevation": 555.86, "height_above_ground": 40},
      {"latitude": 48.088592, "longitude": 11.566361, "ground_elevation": 555.76, "height_above_ground": 40},
      {"latitude": 48.088562, "longitude": 11.566409, "ground_elevation": 555.93, "height_above_ground": 30},
      {"latitude": 48.088524, "longitude": 11.566298, "ground_elevation": 556.13, "height_above_ground": 30}
        ]
      ],
      "list_of_pv_area_information": [
    {"azimuth": 152.65, "tilt": 25, "name": "PV Area 1"}
      ],
      "list_of_ops": [
         {"latitude": 48.088505, "longitude": 11.566374, "ground_elevation": 555.92, "height_above_ground": 30},
         {"latitude": 48.088493, "longitude": 11.566435, "ground_elevation": 555.78, "height_above_ground": 23}
      ],
      "excluded_areas": [],
      "meta_data": {
        "user_id": "123456789",
        "project_id": "123456789",
        "sim_id": "1234f56789",
        "timestamp": 1717787690,
        "utc": 1,
        "project_name": "Downtown house"
      },
      "simulation_parameter": {
        "grid_width": 0.7,
        "resolution": "1min",
        "sun_elevation_threshold": 4,
        "beam_spread": 6.5,
        "sun_angle": 0.5,
        "sun_reflection_threshold": 10.5,
        "zoom_level": 20
      }
    }
    """
    import json
    data = json.loads(example_call)
    
    calculate_glare(
        data["pv_areas"],
        data["list_of_pv_area_information"],
        data["list_of_ops"],
        data["meta_data"],
        data["simulation_parameter"],

        'assets/local',
        data["excluded_areas"],



    )

if __name__ == "__main__":
    test()

    # 1717720922