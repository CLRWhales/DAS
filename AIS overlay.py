#this script computs the ship overlay data for the das cleaner function

#svalbard is in UTM grid 33X
# %%
from shapely.geometry import LineString, Point
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import utm

path = "D:/DAS/cablecoordsUTM/OuterCabel_UTM_Coord_NorgesKart.txt"

data = pd.read_csv(path, sep = "\t")
#data.plot(x = 'UTMX',y = 'UTMY')
data = data.reindex(columns= ['UTMX','UTMY'])
line_coords_UTM = list(data.itertuples(index = False, name = None)) #UTM


line_coords_UTM.reverse() #to move longyearbyen to the top?

line = LineString([utm.to_latlon(east,north,33,'X') for east,north in line_coords_UTM]) #moves UTM to lon,lat and makes a line object

AIS = "D:/DAS/cablecoordsUTM/AIS20220821_22.csv"
track = pd.read_csv(AIS, sep = ';')


track_data = list(track.itertuples(index = False, name = None)) 

# Process each track point
results = []
for lat, lon, timestamp, name, mmsi, imo, callsign in track_data:
    track_point = Point(lon, lat)  # Convert to (x, y) as (lon, lat)
    closest_point = line.interpolate(line.project(track_point))  # Closest point on line
    closest_proportion = line.project(track_point,normalized=True)
    closest_lat, closest_lon = closest_point.y, closest_point.x  # Convert back to (lat, lon)
    
    # Compute geodesic distance (meters)
    distance = geodesic((lat, lon), (closest_lat, closest_lon)).meters
    
    results.append((timestamp, lat, lon, closest_lat, closest_lon,closest_proportion, distance, name,mmsi,imo,callsign))

# Convert results to DataFrame
output = pd.DataFrame(results, columns=["timestamp", "track_lat", "track_lon", "closest_lat", "closest_lon","closest_prop_LYB", "distance_m", "name","MMSI","IMO","Callsing"])

output.to_csv("D:/DAS/cablecoordsUTM/AIS_projection_Outer.csv", index = False)
#print(df)

# %%
