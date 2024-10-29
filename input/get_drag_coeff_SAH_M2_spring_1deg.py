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

    def drag_coeff(lon,lat,omega):
        print('Computing drag coefficient at (%.3f°E,%.3f°N)... \n' % (lon,lat))
        # estimate the mode-1 wavelength
        f_loc = f(lat) # rad/s
        N_loc = get_N_clim(lon,lat) 
        topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')
        depth = topog.z
        H_loc = -depth.interp(lat=lat,lon=lon).values
        delta_lon, delta_lat = get_delta(lat)
        kh = estimate_kh(omega,f_loc,N_loc,H_loc)
        # compute the sampling step in deg based on the wavelength of mode-1 waves
        # step_lon, step_lat = get_radius_in_deg(kh,lon,lat,delta_lon,delta_lat)
        # step = max(step_lon, step_lat)
        # if step < 1:
        #    step = 1
        # if step > 5:
        #    step = 5
        # print('The sampling step is %.1f degree.' % step)
        step = 5        
        # sample the topography
        topog_sample = depth.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)

        # perform spectral analysis
        topog_spd_2d, dx, dy = fft_topog(topog_sample,delta_lon,delta_lat,k_grid_units=False)

        # check Parseval's theorem
        print('Checking Parsevals theorem...')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        fft2dsum = topog_spd_2d.sum().sum()*dkx*dky
        print('2D sum (k-space)     = %3.2e' % fft2dsum)
        var_int_x = np.nansum(topog_sample **2 * dy * dx)
        print('2D sum (x-space)     = %3.2e \n' % var_int_x)

        A_loc = 2 * step * delta_lon * 1e+3 * 2 * step * delta_lat * 1e+3 

        h_rms, k_bar = compute_h_rms(A_loc,topog_spd_2d)
        print('rms height           = %3.2e \n' % h_rms)
        L_bar = 2 * np.pi / k_bar 
        print('height weighted mean k converted scale: %3.2e m \n' % L_bar)

        if np.isnan(kh) == 0 and omega>f_loc and omega<N_loc:
            # compute the drag coefficient
            print('Computing the drag coeff...')
            sigma_xx, sigma_yy, sigma_xy = compute_drag_coeff(A_loc,N_loc,H_loc,omega,f_loc,topog_spd_2d)
            spring_xx, spring_yy, spring_xy = np.nan, np.nan, np.nan
            
        elif np.isnan(kh) == 1 and omega<f_loc:
            # compute the spring force
            print('Computing the spring force...')
            spring_xx, spring_yy, spring_xy = compute_spring_coeff(A_loc,N_loc,H_loc,omega,f_loc,topog_spd_2d)
            sigma_xx, sigma_yy, sigma_xy = np.nan, np.nan, np.nan

        else:
            print('Outside critical latitude.')
            sigma_xx, sigma_yy, sigma_xy, spring_xx, spring_yy, spring_xy = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        print('Drag coefficient (x-dir): %3.2e' % sigma_xx)
        print('Drag coefficient (y-dir): %3.2e' % sigma_yy)
        print('Drag coefficient (cross): %3.2e' % sigma_xy)
        print('Spring force (x-dir): %3.2e' % spring_xx)
        print('Spring force (y-dir): %3.2e' % spring_yy)
        print('Spring force (cross): %3.2e' % spring_xy)

        return sigma_xx, sigma_yy, sigma_xy, spring_xx, spring_yy, spring_xy, h_rms, k_bar

    def get_N_clim(lon,lat):

        Nave_data = xr.open_dataset('/g/data/nm03/lxy581/WOA18/Nave_500m_woa18.nc')

        # Read depth-averaged N and convert the unit to rad/s
        Nave_1km = Nave_data.Nave * 2 * np.pi

        # Use the 2D interpolation to find the depth-averaged N
        N_clim = Nave_1km.interp(lat=lat,lon=lon).values

        return N_clim

    def estimate_kh(omega,f_loc,N_loc,H_loc):

        kh = np.sqrt((omega**2-f_loc**2)*np.pi**2/N_loc**2/H_loc**2)
        print('kh^2 = ', (omega**2-f_loc**2)*np.pi**2/N_loc**2/H_loc**2)

        return kh

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

    def windowing(topog_sample, alpha=0.8, border=0.9):

        # shape parameter of the Tukey window (flat + cosine tapered region)
        # - raidus < alpha, factor = 1
        # - alpha < radius < alpha, factor decays from 1 to 0
        # - radius > border, factor = 0
        
        vari = np.var(topog_sample)

        ydim, xdim = topog_sample.dims
        nx = topog_sample[xdim].size
        ny = topog_sample[ydim].size

        # Create coordinate grid
        y = np.linspace(-1, 1, ny)
        x = np.linspace(-1, 1, nx)
        xv, yv = np.meshgrid(x, y)
        
        # Calculate radial distance from the center
        r = np.sqrt(xv**2 + yv**2)
        
        # Initialize the window with zeros
        window = np.zeros((ny, nx))
        
        # Apply Tukey window formula
        for j in range(ny):
            for i in range(nx):
                if r[j, i] <= alpha:
                    window[j, i] = 1
                elif r[j, i] <= border:
                    window[j, i] = 0.5*(1 + np.cos( np.pi * (r[j, i] - alpha ) / (border - alpha) ))
                else:
                    window[j, i] = 0

        topog_filt = topog_sample * window
        varf = np.var(topog_filt)
        fac  = np.sqrt(vari/varf).values
        window *= fac
        topog_filt *= fac

        return topog_filt

    def fft_topog(topog,delta_lon,delta_lat,alpha=0.8,border=0.9,k_grid_units=True):

        ydim, xdim = topog.dims
        nx = topog[xdim].size
        ny = topog[ydim].size
        dx = np.mean(np.diff(topog[xdim]))*delta_lon*1e+3
        dy = np.mean(np.diff(topog[ydim]))*delta_lat*1e+3

        # demean
        topog -= topog.mean(skipna=True) 

        # windowing
        topog_filt = windowing(topog,alpha=0.8,border=0.9)

        # FFT
        topog_fft = fft2(topog_filt.values)
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

    def compute_drag_coeff(A,N,H,omega,f,topog_spd_2d):

        # *H: integrate in depth so that the final unit is m/s
        const = 1 / (4 * np.pi**2 * H * A) * H

        kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        K_2D = np.sqrt(kx_2D**2 + ky_2D**2)
        m_2D = K_2D * np.sqrt((N**2 - omega**2)/(omega**2 - f**2))

        int_common = topog_spd_2d * (N**2 - omega**2) / (m_2D*omega)
        int_xx = int_common * kx_2D**2 * dkx * dky
        int_yy = int_common * ky_2D**2 * dkx * dky
        int_xy = int_common * kx_2D*ky_2D * dkx * dky
        sigma_xx = const * np.nansum(int_xx[:])
        sigma_yy = const * np.nansum(int_yy[:])
        sigma_xy = const * np.nansum(int_xy[:])

        return sigma_xx, sigma_yy, sigma_xy

    def compute_spring_coeff(A,N,H,omega,f,topog_spd_2d):

        # *H: integrate in depth so that the final unit is m/s
        const = 1 / (4 * np.pi**2 * H * A * omega) * H

        kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        K_2D = np.sqrt(kx_2D**2 + ky_2D**2)
        m_2D = K_2D * np.sqrt((N**2 - omega**2)/(f**2 - omega**2))

        int_common = topog_spd_2d * (N**2 - omega**2) / (m_2D*omega)
        int_xx = int_common * kx_2D**2 * dkx * dky
        int_yy = int_common * ky_2D**2 * dkx * dky
        int_xy = int_common * kx_2D*ky_2D * dkx * dky
        spring_xx = const * np.nansum(int_xx[:])
        spring_yy = const * np.nansum(int_yy[:])
        spring_xy = const * np.nansum(int_xy[:])

        return spring_xx, spring_yy, spring_xy

    def compute_h_rms(A,topog_spd_2d):

        const = 1 / (4 * A * np.pi**2)
        kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        int_kl = topog_spd_2d * dkx * dky
        h_rms = np.sqrt(const * np.nansum(int_kl[:]))

        k_mag = np.sqrt(kx_2D**2 + ky_2D**2)
        int_kl = k_mag * topog_spd_2d * dkx * dky
        k_bar = 1/ h_rms**2 * const * np.nansum(int_kl[:])

        return h_rms, k_bar

    omega_M2 = 2 * np.pi / (12.4206014*3600)
    omega_K1 = 2 * np.pi / (23.9344658*3600)

    itile = int(sys.argv[1])
    ilon = (itile-1)//30+1
    ilat = itile - (ilon-1)*30

    lat_id = np.arange((ilat-1)*9*10,ilat*9*10,10)
    lon_id = np.arange((ilon-1)*12*10,ilon*12*10,10)

    grid = xr.open_dataset('/g/data/nm03/lxy581/archive/tides_01_global_cdrag_const/output002/ocean_static.nc')
    lat = np.array(grid["geolat"].isel(yh=lat_id, xh=lon_id))
    lon = np.array(grid["geolon"].isel(yh=lat_id, xh=lon_id))
    lon[lon < -180] += 360
    yh = np.array(grid["yh"].isel(yh=lat_id))
    xh = np.array(grid["xh"].isel(xh=lon_id))

    ny = lat_id.size
    nx = lon_id.size

    depth = np.full((ny,nx),np.nan)
    sigma_xx = np.full((ny,nx),np.nan)
    sigma_yy = np.full((ny,nx),np.nan)
    sigma_xy = np.full((ny,nx),np.nan)
    spring_xx = np.full((ny,nx),np.nan)
    spring_yy = np.full((ny,nx),np.nan)
    spring_xy = np.full((ny,nx),np.nan)
    h_rms = np.full((ny,nx),np.nan)
    k_bar = np.full((ny,nx),np.nan)

    topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')

    for j in range(ny):
        for i in range(nx):
            depth[j,i] = topog.z.interp(lat=lat[j,i],lon=lon[j,i]).values
            if np.isnan(get_N_clim(lon[j,i],lat[j,i])) == 0 and np.isnan(topog.z.interp(lat=np.arange(lat[j,i]-3,lat[j,i]+3),lon=np.arange(lon[j,i]-3,lon[j,i]+3)).mean(skipna=False)) == 0:
                sigma_xx[j,i], sigma_yy[j,i], sigma_xy[j,i], spring_xx[j,i], spring_yy[j,i], spring_xy[j,i], h_rms[j,i], k_bar[j,i] = drag_coeff(lon[j,i],lat[j,i],omega_M2)

    # saving data

    lat = xr.DataArray(lat, dims=['yh','xh'],
                            coords={'yh': yh,
                                    'xh': xh},
                            attrs={'long_name': 'latitude', 'units': 'degrees_north'})

    lon = xr.DataArray(lon, dims=['yh','xh'],
                            coords={'yh': yh,
                                    'xh': xh},
                            attrs={'long_name': 'longitude', 'units': 'degrees_east'})

    depth = xr.DataArray(depth, dims=['yh','xh'],
                                coords={'yh': yh,
                                        'xh': xh},
                                attrs={'long_name': 'depth', 'units': 'm'})

    sigma_xx_xr = xr.DataArray(sigma_xx, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'x-dir drag coefficient, M2', 'units': 'm second-1'})

    sigma_yy_xr = xr.DataArray(sigma_yy, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'y-dir drag coefficient, M2', 'units': 'm second-1'})

    sigma_xy_xr = xr.DataArray(sigma_xy, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'xy-dir drag coefficient, M2', 'units': 'm second-1'})

    spring_xx_xr = xr.DataArray(spring_xx, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'x-dir drag coefficient, M2', 'units': 'm second-1'})

    spring_yy_xr = xr.DataArray(spring_yy, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'y-dir drag coefficient, M2', 'units': 'm second-1'})

    spring_xy_xr = xr.DataArray(spring_xy, dims=['yh','xh'],
                                         coords={'yh': yh,
                                                 'xh': xh},
                                         attrs={'long_name': 'xy-dir drag coefficient, M2', 'units': 'm second-1'})

    hrms_xr = xr.DataArray(h_rms, dims=['yh','xh'],
                                  coords={'yh': yh,
                                          'xh': xh},
                                  attrs={'long_name': 'rms height, M2', 'units': 'm'})

    kbar_xr = xr.DataArray(k_bar, dims=['yh','xh'],
                                  coords={'yh': yh,
                                          'xh': xh},
                                  attrs={'long_name': 'height-weighted-mean wavenumber, M2', 'units': 'm'})

    sigma = xr.Dataset({"sigma_xx": sigma_xx_xr,
                        "sigma_yy": sigma_yy_xr,
                        "sigma_xy": sigma_xy_xr,
                        "spring_xx": spring_xx_xr,
                        "spring_yy": spring_yy_xr,
                        "spring_xy": spring_xy_xr,
                        "h_rms": hrms_xr,
                        "k_bar": kbar_xr,
                        "lon": lon,
                        "lat": lat,
                        "depth": depth})

    sigma.to_netcdf('/g/data/nm03/lxy581/global_drag_coeff/sigma_SAH_M2_1deg_spring_2d_%03d.nc' % (itile))
