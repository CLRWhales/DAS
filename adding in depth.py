#%% load in the interpolated csv, and the depth csv and try and flip and merge,

import pandas as pd
import xarray as xr
import numpy as np


def extract_nearest_bathymetry_from_csv(netcdf_path, csv_path, lat_col='latitude', lon_col='longitude'):
    """
    Reads a CSV file with latitude and longitude columns, and a GEBCO NetCDF bathymetry file.
    Returns the bathymetry depths at the nearest neighbor grid points for each location.
    
    Args:
        netcdf_path (str): Path to GEBCO NetCDF bathymetry file.
        csv_path (str): Path to CSV file with latitude and longitude columns.
        lat_col (str): Name of the latitude column in the CSV.
        lon_col (str): Name of the longitude column in the CSV.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'depth' column.
    """
    # Read coordinates from CSV
    df = pd.read_csv(csv_path)

    # Open GEBCO dataset
    ds = xr.open_dataset(netcdf_path)
    depth_var = 'elevation' if 'elevation' in ds else list(ds.data_vars)[0]

    # Use xarray's vectorized selection
    depths = ds[depth_var].sel(
        lat=xr.DataArray(df[lat_col], dims='points'),
        lon=xr.DataArray(df[lon_col], dims='points'),
        method='nearest'
    ).values

    # Add depth to the DataFrame
    df['depth'] = depths
    return df


track = "D:\\DAS\\cablecoordsUTM\\interpolated_latlon.csv"
bathy = "D:\\DAS\\Bathy\\GEBCO_07_May_2025_e50f625ee422\\gebco_2024_n81.3977_s75.4797_w8.6096_e27.6379.nc"
output_df = extract_nearest_bathymetry_from_csv(bathy, track)

output_df = output_df[::-1]
channel = np.arange(len(output_df))*4 + 4200
output_df.insert(0,'chan',channel)

output_df = output_df.reindex(columns=['chan','latitude','longitude','depth'])
output_df['depth'] = abs(output_df['depth'])
output_df.to_csv("D:\\DAS\\DASsourceLOC\\Svalbard_DAS_latlondepth_outer.txt", index=False)
# %%
