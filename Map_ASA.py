#this makes a map of the location and adds in tracks and stuff
#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import cmocean
import pandas as pd
import matplotlib.dates as mdates

netcdf_path = "D:\\DAS\\Bathy\\GEBCO_07_May_2025_e50f625ee422\\gebco_2024_n81.3977_s75.4797_w8.6096_e27.6379.nc"
dst = 'C:\\Users\\Calder\\OneDrive - NTNU\\Documents\\Presentations\\ASA May 2025\\Figures\\map1.png'

# Load the dataset
ds = xr.open_dataset(netcdf_path)
depth_var = 'elevation' if 'elevation' in ds else list(ds.data_vars)[0]
bathy = ds[depth_var]



#other data 
path = "D:\\DAS\\cablecoordsUTM\\OuterCabel_UTM_Coord_NorgesKart.txt"
data = pd.read_csv(path, sep = "\t")
#data.plot(x = 'UTMX',y = 'UTMY')
data = data.reindex(columns= ['UTMX','UTMY'])

path = "D:\\DAS\\cablecoordsUTM\\InnerCabel_UTM_Coord_NorgesKart.txt"
data2 = pd.read_csv(path, sep = "\t")
#data.plot(x = 'UTMX',y = 'UTMY')
data2 = data2.reindex(columns= ['UTMX','UTMY'])


path = "D:\\DAS\\cablecoordsUTM\\AIS20220821_22.csv"
data3 = pd.read_csv(path, sep = ";")
names = data3['name'].unique()
data3['date_time_utc'] = pd.to_datetime(data3['date_time_utc'])
data3['timestamp_num'] = mdates.date2num(data3['date_time_utc'])*86400



t = np.load("D:\\DAS\\DASsourceLOC\\t.npy")
tnorm = (t -  data3['timestamp_num'].min())/(data3['timestamp_num'].max() - data3['timestamp_num'].min())  

ship1 = np.load('D:\\DAS\\DASsourceLOC\\ship1.npy', allow_pickle = True)
ship2 = np.load('D:\\DAS\\DASsourceLOC\\ship2.npy',allow_pickle = True)

#%%
start_date = '2022-08-21 18:00:00'
end_date = '2022-08-21 19:00:00'
mask = (data3['date_time_utc'] >= start_date) & (data3['date_time_utc'] <= end_date)
data3 = data3.loc[mask]
data3['timestamp_num'] = (data3['timestamp_num'] - data3['timestamp_num'].min()) / (data3['timestamp_num'].max() - data3['timestamp_num'].min())    



data4 = data3
data3 = data3[data3['name'] == 'VILLA']
data4 = data4[data4['name'] == 'COMMANDANT CHARCOT']

norm2 = plt.Normalize(data3['timestamp_num'].min(), data3['timestamp_num'].max())

whale1 = np.load("D:\\DAS\\DASsourceLOC\\r1.npy")
whale2 = np.load("D:\\DAS\\DASsourceLOC\\r2.npy")
mask3 = np.where(whale1[:,3]<800)
mask4 = np.where(whale2[:,3]>-800)
#%%
# Create the plot
fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.UTM(33))
ax.set_extent([13.5, 18, 78, 78.5], crs=ccrs.PlateCarree())
# Define normalization so that 0 is the midpoint
norm = TwoSlopeNorm(vmin=-400, vcenter=0, vmax=1200)
# Plot the data
img = bathy.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    x='lon',
    y='lat',
    cmap=cmocean.cm.topo,
    norm = norm,
    add_colorbar=True,
    cbar_kwargs={'label': 'Elevation (m)'}
)
plt.plot(data['UTMX'],data['UTMY'], transform = ccrs.UTM(33), color = 'black', linestyle= 'dashed', linewidth = 3)
plt.plot(data2['UTMX'],data2['UTMY'], transform = ccrs.UTM(33), color = 'blue', linestyle= 'dashed', linewidth = 3)
#plt.plot(data3['lat'],data3['lng'], transform = ccrs.PlateCarree(), color = 'blue')
plt.scatter(data3['lat'],data3['lng'],c = data3['timestamp_num'], transform = ccrs.PlateCarree(), cmap = 'plasma')
#plt.scatter(data3.head(1)['lat'],data3.head(1)['lng'], transform = ccrs.PlateCarree(), color = 'red')

plt.scatter(data4['lat'],data4['lng'],c = data4['timestamp_num'], transform = ccrs.PlateCarree(), cmap = 'plasma')

# plt.scatter(ship1[:,2],ship2[:,3], transform = ccrs.PlateCarree(), marker = '+', color = 'pink')
# plt.scatter(ship2[:,2],ship2[:,3], transform = ccrs.PlateCarree(), marker = '+', color = 'pink')


#plt.scatter(data4.head(1)['lat'],data4.head(1)['lng'], transform = ccrs.PlateCarree(), color = 'red')
# for name, group in data4.groupby('name'):
#     plt.plot(group['lat'], group['lng'], label=name,transform = ccrs.PlateCarree())
#plt.scatter(whale1[mask3,2],whale1[mask3,1],c = tnorm[mask3],transform = ccrs.PlateCarree(), cmap = 'plasma', marker = '+')
#plt.scatter(whale2[mask4,2],whale2[mask4,1],c = tnorm[mask4],transform = ccrs.PlateCarree(), cmap = 'plasma', marker = '+')
#plt.colorbar()
# plt.legend(title='ship')
# Add geographic features
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.LAND, facecolor='lightgray')
#ax.set_title("Inner Isfjorden", fontsize=16)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# Save or show

plt.savefig(dst, dpi=500)
plt.show()

# Example usage


# %%
plt.figure()
plt.scatter(ship1[:,2],ship2[:,3], marker = '+', color = 'pink')
plt.scatter(ship2[:,2],ship2[:,3],  marker = '+', color = 'pink')
# %%
