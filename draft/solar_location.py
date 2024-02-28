# # import datetime
# # import pvlib
# # from math import sin, cos , acos , sqrt
# #
# # # Observer's location (latitude, longitude, elevation in meters)
# # latitude = 37.7749  # Latitude of San Francisco, CA
# # longitude = -122.4194  # Longitude of San Francisco, CA
# # elevation = 15.56675434112549  # Elevation in meters
# #
# # # Timezone
# # timezone = 'US/Pacific'
# #
# # # Get current date and time
# # now = datetime.datetime.now()
# #
# # # Create location object
# # location = pvlib.location.Location(latitude, longitude, tz=timezone, altitude=elevation)
# #
# # # Solar position at current time
# # solar_position = location.get_solarposition(now)
# #
# # # Print Sun's altitude and azimuth
# # print("Sun's altitude: ", solar_position['apparent_elevation'])
# # print("Sun's azimuth: ", solar_position['azimuth'])
# # # print(solar_position)
# #
# # sun_azim = solar_position['azimuth']
# # sun_elev = solar_position['apparent_elevation']
# # Vx_s = sin(sun_azim) * cos(sun_elev)
# # Vy_s = cos(sun_azim) * cos(sun_elev)
# # Vz_s = sin(sun_elev)
# #
# # Vx_r = sin(sun_azim + 180) * cos(sun_elev)
# # Vy_r = cos(sun_azim + 180) * cos(sun_elev)
# # Vz_r = sin(sun_elev)
# #
# # X_p =latitude
# # Y_p =longitude
# # Z_p =elevation
# #
# # latitude = 37.7749  # Latitude of San Francisco, CA
# # longitude = -122.4194  # Longitude of San Francisco, CA
# # elevation = 15.56675434112549  # Elevation in meters
# #
# # X_detection_point = 37.774702
# # Y_detection_point = -122.419204
# # Z_detection_point = 15.21727085113525
# #
# # Vx_p = X_detection_point - X_p
# # Vy_p = Y_detection_point - Y_p
# # Vz_p = Z_detection_point - Z_p
# #
# # alpha = acos( (Vx_p*Vx_r + Vy_p*Vy_r + Vz_p*Vz_r) / sqrt(Vx_p**2 + Vy_p**2 + Vz_p**2) )
# #
# # print(alpha)
#
#
#
# import numpy as np
#
# def check_sun_reflection(sun_position, earth_point, target_point, surface_normal):
#     # Vector from the earth point to the sun
#     sun_to_earth = sun_position - earth_point
#
#     # Vector from the earth point to the target point
#     earth_to_target = target_point - earth_point
#
#     # Calculate the angle between the surface normal and the vector from the earth point to the sun
#     angle_of_incidence = np.arccos(np.dot(sun_to_earth, surface_normal) / (np.linalg.norm(sun_to_earth) * np.linalg.norm(surface_normal)))
#
#     # Calculate the angle between the surface normal and the vector from the earth point to the target point
#     angle_of_reflection = np.arccos(np.dot(earth_to_target, surface_normal) / (np.linalg.norm(earth_to_target) * np.linalg.norm(surface_normal)))
#
#     # If the angle of incidence equals the angle of reflection, the sun's reflection will hit the target point
#     return np.isclose(angle_of_incidence, angle_of_reflection)
#
# # Example positions (in 3D space)
# sun_position = np.array([100, 100, 100])
# earth_point = np.array([0, 0, 0])
# target_point = np.array([10, 50, 50])
#
# # Example surface normal (for simplicity, assuming it's pointing directly upwards)
# surface_normal = np.array([0, 0, 1])
#
# # Check if sun's reflection will hit the target point
# will_hit = check_sun_reflection(sun_position, earth_point, target_point, surface_normal)
# print("Will sun's reflection hit the target point?", will_hit)