# from skyfield.api import Topos, load
# from datetime import datetime
#
# # Load the ephemeris data
# eph = load('de421.bsp')
#
# # Define observer location (latitude, longitude, elevation)
# observer = Topos(latitude_degrees=37.7749, longitude_degrees=-122.4194, elevation_m=0)
#
# # Get current time
# ts = load.timescale()
# current_time = ts.now()
#
# # Compute positions of solar system bodies
# sun = eph['sun']
# observer_location = eph['earth'] + observer
# sun_position = observer_location.at(current_time).observe(sun).apparent().position.km
#
# print("Sun's position (x, y, z):", sun_position)
