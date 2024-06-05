import sys
import xrft
import math
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
import climtas.nci

if __name__ == '__main__':
    climtas.nci.GadiClient()

    def drag_coeff(lon,lat):

        print('Computing drag coefficient at (%.1f°E,%.1f°N)... \n' % (lon,lat))
        kh    = 2 * np.pi / 1e+4
        omega = 2 * np.pi / (12.42*3600)
        f_loc = f(lat)
        N_loc = get_N_clim(lon,lat)
        print('Local bottom stratification is %.1e rad/s' % N_loc)
        topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')
        depth = topog.z
        if omega>f_loc and omega<N_loc:
            step = 1
            # sample the topography
            topog_sample = depth.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)

            delta_lon, delta_lat = get_delta(lat)
            topog_spd_2d, dx, dy = fft_topog(topog_sample,delta_lon,delta_lat,k_grid_units=False)

            A_loc = 2 * step * delta_lon * 1e+3 * 2 * step * delta_lat * 1e+3 
            h_rms = compute_h_rms(A_loc,topog_spd_2d)

            sigma = compute_drag_coeff(kh,h_rms,N_loc)

        else:
            print('This grid point is poleward of critical latitude.')
            sigma = np.nan

        print('Drag coefficient: %3.2e' % sigma)

        return sigma

    def get_N_clim(lon,lat):

        Nbot_data = xr.open_dataset('/g/data/nm03/lxy581/WOA18/Nbot_1000m_woa18.nc')

        # Read bottom N and convert the unit to rad/s
        Nbot_1km = Nbot_data.Nbot * 2 * np.pi

        # Use the 2D interpolation to find the bottom N
        N_clim = Nbot_1km.interp(lat=lat,lon=lon).values

        return N_clim

    def get_delta(lat):

        # compute the distance in km at this lon and lat
        delta_lon = 2 * np.pi * (6371 * np.cos(lat*np.pi/180)) / 360
        delta_lat = 1 * np.pi * 6371 / 180

        return delta_lon, delta_lat

    def filtering(topog_sample):

        ydim, xdim = topog_sample.dims
        nx = topog_sample[xdim].size
        ny = topog_sample[ydim].size
        win_2d = np.full((ny,nx), np.nan)
        radi = topog_sample.shape[0] / 2
        for i in range(topog_sample.shape[1]):
          for j in range(topog_sample.shape[0]):
            win_2d[j,i] = 1 - ( ((i-radi)/radi)**2 + ((j-radi)/radi)**2 )
        win_2d[win_2d<0]=0

        fac = np.sqrt(win_2d.size / np.sum(win_2d**2))
        win_2d = win_2d * fac
        win_2d_xr = xr.DataArray(win_2d, coords={'y': np.arange(ny), 'x': np.arange(nx)}, dims=["y", "x"])

        topog_filt = np.array(topog_sample) * np.array(win_2d_xr)
        topog_filt = xr.DataArray(topog_filt, coords=topog_sample.coords, dims=topog_sample.dims)

        return topog_filt

    def fft_topog(topog,delta_lon,delta_lat,k_grid_units=True):

        ydim, xdim = topog.dims
        nx = topog[xdim].size
        ny = topog[ydim].size
        dx = np.mean(np.diff(topog[xdim]))*delta_lon*1e+3
        dy = np.mean(np.diff(topog[ydim]))*delta_lat*1e+3

        # demean
        topog -= topog.mean(skipna=True)

        # windowing
        topog_filt = filtering(topog)

        # FFT
        topog_fft = fft2(topog.values)
        topog_spd = (topog_fft*topog_fft.conjugate()).real
        topog_spd[0,0] = np.nan          # nan at removed zero frequency
        topog_spd = fftshift(topog_spd)  # put zero wavenumber in array centre
        topog_spd *= dx*dy

        if (k_grid_units):
            # k in units cycles/dx:
            topog_spd = xr.DataArray(topog_spd, dims=['ky','kx'],
                                     coords={'ky': np.linspace(-0.5, 0.5+(ny%2-1)/ny, num=ny),
                                             'kx': np.linspace(-0.5, 0.5+(nx%2-1)/nx, num=nx)},
                                     attrs={'long_name': 'wavenumber spectrum in grid units'})
            topog_spd.kx.attrs['units'] = 'cycles/dx'
            topog_spd.ky.attrs['units'] = 'cycles/dy'
            topog_spd.kx.attrs['long_name'] = 'x wavenumber'
            topog_spd.ky.attrs['long_name'] = 'y wavenumber'

            # No rescaling

        else:
            # k in units cycles/(units of dx)
            topog_spd = xr.DataArray(topog_spd, dims=['ky','kx'],
                                     coords={'ky': np.linspace(-0.5, 0.5+(ny%2-1)/ny, num=ny)*(2*np.pi/dy),
                                             'kx': np.linspace(-0.5, 0.5+(nx%2-1)/nx, num=nx)*(2*np.pi/dx)},
                                     attrs={'long_name': 'wavenumber spectrum in grid units'})
            topog_spd.kx.attrs['units'] = 'radians/meters'
            topog_spd.ky.attrs['units'] = 'radians/meters'
            topog_spd.kx.attrs['long_name'] = 'x wavenumber'
            topog_spd.ky.attrs['long_name'] = 'y wavenumber'

            # Rescale to satisfy Parseval's theorem:
            topog_spd /= 4*np.pi**2./dx/dy

        return topog_spd, dx, dy

    def compute_h_rms(A,topog_spd_2d):

        const = 1 / (4 * A * np.pi**2)
        kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        int_kl = topog_spd_2d * dkx * dky
        h_rms = np.sqrt(const * np.nansum(int_kl[:]))
        return h_rms

    def compute_drag_coeff(kh,h_rms,N):

        sigma = 0.5 * kh * h_rms**2 * N

        return sigma

    itile = int(sys.argv[1]) 
    ilon = (itile-1)//20+1 
    ilat = itile - (ilon-1)*20
    hres = 1.0 # 0.25

    lat = np.arange(-89.875+(ilat-1)*9,-89.875+ilat*9,hres)
    lon = np.arange(-179.875+(ilon-1)*18,-179.875+ilon*18,hres)

    ny = lat.size
    nx = lon.size

    sigma_itile = np.full((ny,nx),np.nan)

    topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')

    for j in range(ny):
        for i in range(nx):
            if np.isnan(get_N_clim(lon[i],lat[j])) == 0 and np.isnan(topog.z.interp(lat=np.arange(lat[j]-3,lat[j]+3),lon=np.arange(lon[i]-3,lon[i]+3)).mean(skipna=False)) == 0:
                sigma_itile[j,i] = drag_coeff(lon[i],lat[j])

    # saving data

    lat = xr.DataArray(lat, dims=['lat'],
                            coords={'lat': lat},
                            attrs={'long_name': 'latitude', 'units': 'degrees_north'})

    lon = xr.DataArray(lon, dims=['lon'],
                            coords={'lon': lon},
                            attrs={'long_name': 'longitude', 'units': 'degrees_east'})

    sigma_JSL = xr.DataArray(sigma_itile, dims=['lat','lon'],
                                          coords={'lat': lat,
                                                  'lon': lon},
                                          attrs={'long_name': 'y-dir drag coefficient', 'units': 'second-1'})

    sigma = sigma_JSL.to_dataset(name='sigma_JSL')
    sigma['lat'] = lat
    sigma['lon'] = lon

    sigma.to_netcdf('/g/data/nm03/lxy581/global_drag_coeff/sigma_JSL_2d_%03d.nc' % (itile))
    
