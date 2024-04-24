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
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

def geographic_to_cartesian(points, origin):
    cartesian_coordinates = []
    origin_lat = origin['latitude']
    origin_lon = origin['longitude']
    origin_elevation = origin['ground_elevation'] + origin['height_above_ground']
    for point in points:
        delta_lat = (point['latitude'] - origin_lat) * 111319.9
        delta_lon = (point['longitude'] - origin_lon) * 111319.9 * math.cos(math.radians(point['latitude']))
        delta_elevation = (point['ground_elevation'] + point['height_above_ground']) - origin_elevation
        cartesian_coordinates.append((delta_lon, delta_lat, delta_elevation))
    return cartesian_coordinates

def fit_plane_to_points(points):
    """
    Berechnet die Best-Fit-Ebene durch die Punkte mit der Methode der kleinsten Quadrate.
    :param points: Eine Liste von Punkten [(x1, y1, z1), (x2, y2, z2), ...]
    :return: Die Koeffizienten der Ebene (A, B, C, D), wo die Ebene durch Ax + By + Cz + D = 0 definiert ist.
    """
    # Extrahiere die Koordinaten der Punkte
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    # Berechne die Koeffizienten der Ebene mit der Methode der kleinsten Quadrate
    A, B, C, D = lstsq([x, y, [1]*len(points)], z)[0]

    return A, B, C, D

def generate_calculation_points(points, width):
    points = np.array(points)
    pca = PCA(n_components=2)  # Annahme: die Punkte liegen annähernd in einer Ebene
    points_transformed = pca.fit_transform(points)
    hull = ConvexHull(points_transformed)

    # Generiere ein Gitter über die Bounding Box der konvexen Hülle
    x_min, x_max = points_transformed[:, 0].min(), points_transformed[:, 0].max()
    y_min, y_max = points_transformed[:, 1].min(), points_transformed[:, 1].max()

    x_coords = np.arange(x_min, x_max + width, width)
    y_coords = np.arange(y_min, y_max + width, width)
    grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Filtere die Punkte, die innerhalb der konvexen Hülle liegen
    grid = np.array([point for point in grid if is_point_inside_hull(point, hull)])

    # Transformiere zurück in das globale Koordinatensystem
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
    
    # Koordinaten entsprechend den geografischen Konventionen zuweisen
    scatter = ax.scatter(cps_array[:, 0], cps_array[:, 1], cps_array[:, 2], c=colors, cmap='viridis', label='CP')
    
    # Korrektur für das Zeichnen der Beobachtungspunkte und des Polygons, wenn notwendig
    for op in op_transformed:
        ax.scatter(op[0], op[1], op[2], color='red', marker='x', label='OP')
    
    pv_points_array = np.array(pv_points_transformed + [pv_points_transformed[0]], dtype='float')
    ax.plot(pv_points_array[:, 0], pv_points_array[:, 1], pv_points_array[:, 2], color='green', linestyle='-', linewidth=2, label='PV-Feld')
    
    # Anpassung der Achsenbeschriftungen entsprechend der tatsächlichen Koordinaten
    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')
    ax.set_zlabel('Height (Z)')
    
    # Umkehrung der Y-Achse, um die Plotrichtung zu ändern
    ax.set_ylim(ax.get_ylim()[::-1])
    
    ax.set_title(f'Result for OP {op_index + 1}')
    plt.colorbar(scatter, ax=ax, label='Minutes of glare per year')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_sun_positions(origin_point, year, timezone_offset, delta_t='1min', sun_threshold=0):
    # Erstellen eines Location-Objekts mit Höhe über dem Meeresspiegel
    site_location = Location(latitude=origin_point['latitude'],
                             longitude=origin_point['longitude'],
                             tz=f"Etc/GMT{'+' if timezone_offset < 0 else '-'}{abs(timezone_offset)}",
                             altitude=origin_point['ground_elevation'] + origin_point['height_above_ground'])
    
    # Erstellen des Zeitbereichs für das angegebene Jahr
    times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:59', freq=delta_t, tz=site_location.tz)
    
    # Berechnen der Sonnenpositionen für den Standort und den Zeitbereich
    solpos = site_location.get_solarposition(times)
    
    # Filtern der Sonnenpositionen basierend auf der Sonnenhöhe über dem Horizont
    filtered_solpos = solpos[solpos['apparent_elevation'] > sun_threshold]
    
    # Umbenennen der 'elevation'-Spalte in 'sun_height'
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

    dataframes_list = []  # Liste zur Speicherung der DataFrames für jeden op_index

    for pv_area_index, (reflected_vecs, vecs_op_to_cps) in enumerate(zip(reflected_vecs_list, vecs_op_to_cps_list)):
        pv_area_info = pv_area_information[pv_area_index]

        for op_index, (op_vec, op_to_cp_vec) in enumerate(zip(reflected_vecs, vecs_op_to_cps)):
            op_records = []  # Sammle Datensätze spezifisch für diesen op_index
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
                # Erstelle einen DataFrame für den aktuellen op_index und füge ihn zur Liste hinzu
                op_df = pd.DataFrame(op_records)
                dataframes_list.append(op_df)

    return dataframes_list

def plot_sun_position_reflections_date_time(df, utc, i):
    # Konvertierung der Timestamps in das richtige Format und Entfernung von Duplikaten
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Aggregiere Daten, um für jeden Zeitstempel nur einen Zustand zu haben
    # Priorisierung von False für 'superpositioned_by_sun', wenn gemischte Zustände vorhanden sind
    aggregated_df = df.groupby('timestamp')['superpositioned_by_sun'].agg(lambda x: False if False in x.values else True).reset_index()
    
    # Farbe basierend auf dem aggregierten 'superpositioned_by_sun'
    colors = aggregated_df['superpositioned_by_sun'].map({True: 'gray', False: 'gold'}).values
    
    # Umwandlung der Zeit in Dezimalstunden für das Plotten
    decimal_hours = aggregated_df['timestamp'].dt.hour + aggregated_df['timestamp'].dt.minute / 60.0

    # Erstelle den Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter Plot
    sc = ax.scatter(aggregated_df['timestamp'], decimal_hours, c=colors, alpha=0.75, s=10, marker='s')

    # Legende
    legend_labels = {'gray': 'Sonne überlagert Reflexion', 'gold': 'Blendwirkung'}
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=label) 
               for color, label in legend_labels.items()]
    ax.legend(handles=handles, loc='upper left')

    # Formatierung der X-Achse
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.invert_yaxis()

    # Setze Achsenbeschriftungen
    ax.set_xlabel('Datum')
    ax.set_ylabel('Uhrzeit (UTC+'+str(utc)+')')
    ax.set_title('Detected glare periods for OP' + str(i+1))

    # Anpassung der Y-Achse Formatierung
    def time_formatter(x, pos):
        hours = int(x)
        minutes = int((x - hours) * 60)
        return f'{hours:02d}:{minutes:02d}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(time_formatter))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_geometry(pv_points_transformed_list, cps_list, ops_transformed):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Farben für die verschiedenen Gruppen
    pv_colors = ['green', 'lightgreen', 'darkgreen', 'lime', 'olive']
    cp_colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'white']
    
    # Zeichne die PV-Polygone für jede Gruppe
    for i, pv_points_transformed in enumerate(pv_points_transformed_list):
        pv_points_array = np.array(pv_points_transformed)
        pv_polygon = np.vstack([pv_points_array, pv_points_array[0]])
        ax.plot(pv_polygon[:, 0], pv_polygon[:, 1], pv_polygon[:, 2], color=pv_colors[i % len(pv_colors)], linestyle='-', linewidth=2, label=f'PV-Polygon Gruppe {i+1}')
    
    # Zeichne die Berechnungspunkte (cps) für jede Gruppe
    for i, cps in enumerate(cps_list):
        cps_array = np.array(cps)
        ax.scatter(cps_array[:, 0], cps_array[:, 1], cps_array[:, 2], color=cp_colors[i % len(cp_colors)], s=50, label=f'Berechnungspunkte Gruppe {i+1}')
    
    # Konvertiere Liste in NumPy-Array für eine einfachere Handhabung
    ops_array = np.array(ops_transformed)
    # Zeichne die Beobachtungspunkte (ops)
    ax.scatter(ops_array[:, 0], ops_array[:, 1], ops_array[:, 2], color='red', s=100, marker='x', label='Beobachtungspunkte (OP)')
    
    # Anpassung der Achsenbeschriftungen und Hinzufügen einer Legende
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Titel für den Plot
    ax.set_title('Visualisierung von PV-Polygonen, CPs und OPs')
    
    plt.tight_layout()
    plt.show()

def save_df_to_excel_with_timestamp(df):
    # Erstelle eine Kopie des DataFrames, um die Originaldaten unverändert zu lassen
    df_copy = df.copy()
    
    # Entferne die Zeitzoneninformationen aus allen Datetime-Spalten in der Kopie
    for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        df_copy[col] = df_copy[col].dt.tz_localize(None)

    # Generiere den aktuellen Zeitstempel
    current_time = datetime.datetime.now()
    
    # Formatierung des Zeitstempels für die Verwendung im Dateinamen
    timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Erstelle den Dateinamen mit dem Zeitstempel
    filename = f'data_{timestamp_str}.xlsx'
    
    # Speichere die Kopie des DataFrame (ohne Zeitzoneninformationen) in eine Excel-Datei
    df_copy.to_excel(filename, index=False)
    
    print(f'DataFrame wurde gespeichert unter: {filename}')

def clean_and_simplify_dfs(result_dfs):
    simplified_dfs = []
    for df in result_dfs:
        # Entferne alle Spalten außer 'timestamp' und 'superpositioned_by_sun'
        simplified_df = df[['timestamp', 'superpositioned_by_sun']].copy()
        
        # Gruppiere nach 'timestamp' und wende die spezifische Logik für 'superpositioned_by_sun' an
        simplified_df = simplified_df.groupby('timestamp')['superpositioned_by_sun'].agg(lambda x: False if False in x.values else True).reset_index()
        
        simplified_dfs.append(simplified_df)
    
    return simplified_dfs

def create_daily_summary_dfs(simplified_dfs):
    daily_summary_dfs = []
    for df in simplified_dfs:
        # Datum ohne Zeit für die Gruppierung extrahieren
        df['date'] = df['timestamp'].dt.date
        
        # Gruppiere nach 'date' und zähle die Einträge pro Tag
        daily_summary = df.groupby('date').agg(
            glare=('superpositioned_by_sun', lambda x: (~x).sum()),  # Zählt, wie oft False vorkommt
            superpositioned_by_sun=('superpositioned_by_sun', 'sum')  # Zählt, wie oft True vorkommt
        ).reset_index()
        
        daily_summary_dfs.append(daily_summary)
        
    return daily_summary_dfs


def process_data(pv_areas, list_of_pv_area_information, list_of_ops, utc):
    # SIMULATION PARAMETER
    #######################
    grid_width = 0.25  # Beispiel für Gitterbreite
    year = 2024
    resolution = '1min'
    sun_elevation_threshold = 3
    threshold = 1.2  # Schwellenwert in Grad
    sun_reflection_angle_threshold = 10.49
    #######################

    origin_point = list_of_ops[0]

    # Schritt 1: Transformiere geografische Koordinaten in kartesische Koordinaten
    pv_areas_tansformed = []
    for pv_area in pv_areas:
        pv_area_transformed = geographic_to_cartesian(pv_area, origin_point)
        pv_areas_tansformed.append(pv_area_transformed)
    
    ops_transformed = geographic_to_cartesian(list_of_ops, origin_point)

    # Schritt 2: Generiere Berechnungspunkte
    cps_set = []
    for pv_area_transformed in pv_areas_tansformed:
        cps = generate_calculation_points(pv_area_transformed, grid_width)
        cps_set.append(cps)

    # Schritt 3: Berechne Sonnenpositionen für das Jahr 2024
    sun_positions = calculate_sun_positions(origin_point, year, utc, resolution, sun_elevation_threshold)
    
    # plot_geometry(pv_areas_tansformed, cps_set, ops_transformed)

    # Schritt 4: Berechne 3D-Richtungsvektoren
    set_of_vecs_op_to_cps = []
    for cps in cps_set:
        vecs_op_to_cps = calculate_3d_direction_vectors(ops_transformed, cps)
        set_of_vecs_op_to_cps.append(vecs_op_to_cps)
 
    # Schritt 5: Berechne reflektierte Vektoren
    set_of_reflected_vecs = []
    for i, vecs_op_to_cps in enumerate(set_of_vecs_op_to_cps):
        azimuth, tilt = list_of_pv_area_information[i]['azimuth'], list_of_pv_area_information[i]['tilt']
        reflected_vecs = calculate_reflected_vectors(vecs_op_to_cps, azimuth, tilt)
        set_of_reflected_vecs.append(reflected_vecs)
 
    # Schritt 6: Überprüfe die Richtung der reflektierten Vektoren effizient
    result_dfs = check_reflection_direction_efficient(set_of_reflected_vecs, sun_positions, set_of_vecs_op_to_cps, threshold, sun_reflection_angle_threshold, list_of_pv_area_information)

    # save_df_to_excel_with_timestamp(result_dfs[0])


    for i, df in enumerate(result_dfs):
    #    plot_results_for_op(cps, [ops_transformed[i]], pv_areas_tansformed[i], df, i)
        plot_sun_position_reflections_date_time(df, utc, i)

    # Bereinigung und Vereinfachung der DataFrames
    simplified_dfs = clean_and_simplify_dfs(result_dfs)
    print(simplified_dfs)

    # Erstellung der Tageszusammenfassung
    return simplified_dfs

if __name__ == '__main__':

    # EXAMPLE DATA
    ######################
    # Beispielwerte für PV-Punkte und Beobachtungspunkte #
    pv_point_01 = {"latitude": 53.590556, "longitude": 10.078993, "ground_elevation": 300, "height_above_ground": 15}
    pv_point_02 = {"latitude": 53.590567, "longitude": 10.079046, "ground_elevation": 300, "height_above_ground": 15}
    pv_point_03 = {"latitude": 53.590532, "longitude": 10.079144, "ground_elevation": 300, "height_above_ground": 10}
    pv_point_04 = {"latitude": 53.590488, "longitude": 10.078950, "ground_elevation": 300, "height_above_ground": 10}
    op_01 = {"latitude": 53.590711, "longitude": 10.078924, "ground_elevation": 300, "height_above_ground": 10}
    #######################

    # FROM FRONTEND
    #######################
    # DataSet input from frontend
    pv_areas = [[pv_point_01, pv_point_02, pv_point_03, pv_point_04]]
    list_of_pv_area_information = [{"azimuth": 180, "tilt": 20, "name": "Vertical PV area 1"}]
    list_of_ops = [op_01]
    utc = 1
    #######################
    process_data(pv_areas, list_of_pv_area_information, list_of_ops, utc)




