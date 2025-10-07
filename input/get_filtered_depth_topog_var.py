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
from matplotlib.colors import LogNorm
from scipy.fftpack import fft2, fftshift
from scipy.io import savemat

def get_depth_mean_var(lon,lat,omega,topog):
    print('Computing mean depth and depth var at (%.3f°E,%.3f°N)...' % (lon,lat))
    # estimate the mode-1 wavelength
    f_loc = f(lat) # rad/s
    N_loc = get_N_clim(lon,lat) 
    depth = topog.z
    H_loc = -depth.interp(lat=lat,lon=lon).values
    delta_lon, delta_lat = get_delta(lat)
    
    if omega>np.abs(f_loc) and omega<N_loc:
        # print('Computing within the turning lat...')
        kh2 = estimate_kh2(omega,f_loc,N_loc,H_loc)
        
        # compute the sampling step in deg based on the wavelength of mode-1 waves
        step_lon, step_lat = get_radius_in_deg(np.sqrt(np.abs(kh2)),lon,lat,delta_lon,delta_lat)
        step = max(step_lon, step_lat)
        if step < 0.1:
           step = 0.1
        # print('The sampling step is %.1f degree.' % step)
        
        # sample the topography
        topog_sample = depth.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)
        depth_mean = -np.nanmean(topog_sample)

        # filter the topography with the mode-1 wavelength
        topog_filt = windowing_gaussian(topog_sample, sigma=0.5)
        depth_mean_filt = -np.nanmean(topog_filt)

        topog_prime = topog_sample - topog_filt
        depth_var = np.var(windowing_gaussian(topog_prime, sigma=0.5))
        
    else:
        # print('Outside critical latitude.')
        depth_mean, depth_mean_filt, depth_var = np.nan, np.nan, np.nan
        
    # print('depth_mean: %3.2e' % depth_mean)
    # print('depth_var: %3.2e \n' % depth_var)
    
    return depth_mean, depth_mean_filt, depth_var

def get_N_clim(lon,lat):

    Nave_data = xr.open_dataset('/g/data/nm03/lxy581/WOA18/Nave_500m_woa18.nc')

    # Read depth-averaged N and convert the unit to rad/s
    Nave_1km = Nave_data.Nave * 2 * np.pi

    # Use the 2D interpolation to find the depth-averaged N
    N_clim = Nave_1km.interp(lat=lat,lon=lon).values

    return N_clim

def estimate_kh2(omega,f_loc,N_loc,H_loc):

    kh2 = (omega**2-f_loc**2)*np.pi**2/N_loc**2/H_loc**2
    print('kh^2 = ', kh2)

    return kh2

def get_delta(lat):

    # compute the distance in km at this lon and lat
    delta_lon = 2 * np.pi * (6371 * np.cos(lat*np.pi/180)) / 360
    delta_lat = 1 * np.pi * 6371 / 180

    return delta_lon, delta_lat

def get_radius_in_deg(kh,lon,lat,delta_lon,delta_lat):

    radius = 2 * np.pi / kh / 1000 / 2

    step_lon  = round(radius / delta_lon, 1)
    step_lat  = round(radius / delta_lat, 1)

    return step_lon, step_lat

def windowing_gaussian(topog_sample, sigma=0.5):

    """
    Apply a circular Gaussian window to the sampled topography.
    
    Parameters
    ----------
    topog_sample : xarray.DataArray
        2D sampled topography
    sigma : float
        Standard deviation of the Gaussian (in normalized coordinates [-1,1])
    """
    
    vari = np.var(topog_sample)

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
    window = np.exp(-(r**2) / (2 * sigma**2))

    topog_filt = topog_sample * window
    varf = np.var(topog_filt)
    fac  = np.sqrt(vari/varf).values
    window *= fac
    topog_filt *= fac

    return topog_filt

# print('Starting...')
omega_M2 = 2 * np.pi / (12.4206014*3600)
omega_K1 = 2 * np.pi / (23.9344658*3600)

itile = int(sys.argv[1])
ilon = (itile-1)//30+1 
ilat = itile - (ilon-1)*30
hres = 1.0 # 0.25

lat = np.arange(-89+(ilat-1)*6,-89+ilat*6,hres)
lon = np.arange(-180+(ilon-1)*12,-180+ilon*12,hres)
# print('Constructed lon lat grid...')

start_time = time.time()

ny = lat.size
nx = lon.size

depth_mean_filt = np.full((ny,nx),np.nan)
depth_mean = np.full((ny,nx),np.nan)
depth_var = np.full((ny,nx),np.nan)

topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')

# print('Looping...')
for j in range(ny):
    print('j = ', j)
    for i in range(nx):
        if np.isnan(get_N_clim(lon[i],lat[j])) == 0:
            depth_mean[j,i], depth_mean_filt[j,i], depth_var[j,i] = get_depth_mean_var(lon[i],lat[j],omega_M2,topog)

# print('Saving data...')
# saving data
savemat("/g/data/nm03/lxy581/global_drag_coeff/sigma_SAH_M2_1deg_barotropic_stress_2d_%03d.mat"% (itile), {
"lon_out": lon,
"lat_out": lat,
"depth_var": depth_var,
"mean_depth": depth_mean,
"mean_depth_filt":depth_mean_filt})

end_time = time.time()
exe_time = float(end_time - start_time)
print("Execution time: %.1f mins!" % (exe_time/60.0))
    
