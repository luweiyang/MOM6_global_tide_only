import sys
import xrft
import math
import time
import scipy.io
import numpy as np
import xarray as xr
import netCDF4 as nc
import cmocean as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from gsw import f
from scipy.io import savemat
from matplotlib.colors import LogNorm
from scipy.fftpack import fft2, fftshift

def get_depth_mean_var(lon,lat,omega,topog,N_loc,H_loc,f_loc):
    # print('Computing mean depth and depth var at (%.3f°E,%.3f°N)...' % (lon,lat))
    # estimate the mode-1 wavelength
    delta_lon = 2 * np.pi * (6371 * np.cos(lat*np.pi/180)) / 360
    delta_lat = 1 * np.pi * 6371 / 180
    
    if omega>np.abs(f_loc) and omega<N_loc:
        # print('Computing within the turning lat...')
        
        # compute the sampling step in deg based on the wavelength of mode-1 waves
        step_lon, step_lat = get_radius_in_deg(N_loc,omega,f_loc,H_loc,delta_lon,delta_lat)
        step = max(step_lon, step_lat)
        if step > 5:
            step = 5
        # print('The sampling step is %.1f degree.' % step)
        
        # sample the topography
        # topog_sample = topog.z.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)
        topog_sample = -topog.elevation.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)
        topog_sample = xr.where(topog_sample < 0, 0, topog_sample)

        # filter the topography with the mode-1 wavelength
        topog_filt, topog_mean = windowing_gaussian(topog_sample)
        depth_mean = topog_mean

        topog_prime = (topog_sample - topog_mean)**2
        topog_prime_filt, depth_var = windowing_gaussian(topog_prime)
        
    else:
        # print('Outside critical latitude.')
        depth_mean, depth_var = np.nan, np.nan
        
    # print('depth_mean: %3.2e' % depth_mean)
    # print('depth_var: %3.2e \n' % depth_var)
    
    return depth_mean, depth_var


def get_radius_in_deg(N_loc,omega,f_loc,H_loc,delta_lon,delta_lat):

    radius = np.array(2* np.sqrt((N_loc**2-omega**2)*np.heaviside(N_loc**2-omega**2,0)/np.abs(omega**2-f_loc**2))) *H_loc / 1000 / 2
    
    step_lon  = round(radius / delta_lon, 1)
    step_lat  = round(radius / delta_lat, 1)

    return step_lon, step_lat

def windowing_gaussian(topog_sample):

    ydim, xdim = topog_sample.dims
    nx = topog_sample[xdim].size
    ny = topog_sample[ydim].size

    # Create normalized coordinates
    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    xv, yv = np.meshgrid(x, y)
    
    # Radial distance from the center
    r = np.sqrt(xv**2 + yv**2)
    
    # Gaussian window
    window = np.exp(-(r**2)) 

    topog_filt = topog_sample * window
    topog_filt_ave = np.nansum(topog_filt)/np.nansum(window)

    return topog_filt, topog_filt_ave

# print('Starting...')
omega_M2 = 2 * np.pi / (12.4206014*3600)
omega_K1 = 2 * np.pi / (23.9344658*3600)

itile = int(sys.argv[1])
ilon = (itile-1)//30+1 
ilat = itile - (ilon-1)*30
hres = 1

lat = np.arange(-89+(ilat-1)*6,-89+ilat*6,hres)
lon = np.arange(-180+(ilon-1)*12,-180+ilon*12,hres)
# print('Constructed lon lat grid...')

ny = lat.size
nx = lon.size

depth_mean = np.full((ny,nx),np.nan)
depth_var = np.full((ny,nx),np.nan)

Nave_data = xr.open_dataset('/g/data/nm03/lxy581/WOA18/Nave_500m_woa18.nc')
# Read depth-averaged N and convert the unit to rad/s
Nave_1km = Nave_data.Nave * 2 * np.pi
N_loc_tile = Nave_1km.interp(lat=lat, lon=lon)

# topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')
# H_loc_tile = -topog.z.interp(lat=lat,lon=lon)
topog = xr.open_dataset('/g/data/nm03/lxy581/gebco_2025_sub_ice_topo/GEBCO_2025_sub_ice.nc')
H_loc_tile = -topog.elevation.interp(lat=lat,lon=lon)

f_lat = f(lat) # rad/s

start_time = time.time()

# print('Looping...')
for j in range(ny):
    if j % 5 == 0:
        print(f"Progress: {j}/{ny} rows done")
    for i in range(nx):
        N_loc = N_loc_tile.sel(lat=lat[j], lon=lon[i]).item()
        H_loc = H_loc_tile.sel(lat=lat[j], lon=lon[i]).item()
        if np.isnan(N_loc) == 0:
            depth_mean[j,i], depth_var[j,i] = get_depth_mean_var(lon[i],lat[j],omega_M2,topog,N_loc,H_loc,f_lat[j])

# print('Saving data...')
# saving data
savemat("/g/data/nm03/lxy581/global_drag_coeff/sigma_SAH_M2_1deg_depth_2d_%03d_gebco.mat"% (itile), {
"lon_out": lon,
"lat_out": lat,
"depth_var": depth_var,
"mean_depth": depth_mean})

end_time = time.time()
exe_time = float(end_time - start_time)
print("Execution time: %.1f mins!" % (exe_time/60.0))
    
