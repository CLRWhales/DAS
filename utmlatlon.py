import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev
from pyproj import Transformer

def load_utm_file(filename):
    # Load file, skip the header row
    try:
        coords = np.loadtxt(filename, delimiter=None, skiprows=1)
    except:
        coords = np.loadtxt(filename, delimiter=',', skiprows=1)

    if coords.shape[1] < 2:
        raise ValueError("File must have at least two columns (UTMY, UTMX)")
    
    coords = coords[:, [1, 0]]  # Swap (UTMY, UTMX) → (UTMX, UTMY)
    return coords

def interpolate_utm_track(coords, spacing=4.0):
    tck, u = splprep([coords[:, 0], coords[:, 1]], s=0)
    line = LineString(coords)
    total_length = line.length
    num_points = int(np.ceil(total_length / spacing)) + 1
    distances = np.linspace(0, 1, num_points)
    interpolated = splev(distances, tck)
    return list(zip(interpolated[0], interpolated[1]))

def convert_utm_to_latlon(utm_points, utm_zone=33, northern_hemisphere=True):
    epsg_code = 32600 + utm_zone if northern_hemisphere else 32700 + utm_zone
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    return [transformer.transform(x, y) for x, y in utm_points]

def save_to_csv(points_latlon, filename="interpolated_latlon.csv"):
    with open(filename, "w") as f:
        f.write("longitude,latitude\n")
        for lon, lat in points_latlon:
            f.write(f"{lon:.8f},{lat:.8f}\n")

if __name__ == "__main__":
    input_file = "D:\\DAS\\cablecoordsUTM\\OuterCabel_UTM_Coord_NorgesKart.txt"  # Change this to your actual file
    utm_zone = 33                 # 🔁 Update this to your UTM zone
    northern = True               # 🔁 Set to False if in Southern Hemisphere

    coords = load_utm_file(input_file)
    interpolated_utm = interpolate_utm_track(coords, spacing=4.0)
    interpolated_latlon = convert_utm_to_latlon(interpolated_utm, utm_zone, northern)
    save_to_csv(interpolated_latlon)

    print("✅ Interpolated lat/lon points saved to 'interpolated_latlon.csv'")
