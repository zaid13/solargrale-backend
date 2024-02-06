import math

def spherical_to_normalized_cartesian(theta, phi):
    """
    Convert spherical coordinates to normalized Cartesian coordinates.

    Parameters:
    - theta: Azimuthal angle in radians.
    - phi: Polar angle in radians.

    Returns:
    - x, y, z: Normalized Cartesian coordinates.
    """
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return x, y, z

# Example usage
theta = math.radians(45) # Convert 45 degrees to radians
phi = math.radians(60) # Convert 60 degrees to radians

x, y, z = spherical_to_normalized_cartesian(theta, phi)
print(f"Normalized Cartesian coordinates: x={x}, y={y}, z={z}")