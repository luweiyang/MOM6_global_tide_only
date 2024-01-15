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

def drag_coeff(lon,lat,omega):

    print('Computing drag coefficient at (%.3f°E,%.3f°N)... \n' % (lon,lat))
    # estimate the mode-1 wavelength
    f_loc = f(lat) # rad/s
    N_loc = get_N_clim(lon,lat)
    topog = xr.open_dataset('/g/data/nm03/lxy581/synbath/SYNBATH.nc')
    depth = topog.z
    H_loc = -depth.interp(lat=lat,lon=lon).values
    kh = estimate_kh(omega,f_loc,N_loc,H_loc)
    if np.isnan(kh) == 0 and omega>f_loc and omega<N_loc:
        #print('The horizontal wavelength of the mode-1 M2 internal tides is approx. %d km. \n' % (2 * np.pi / kh / 1000))


        # compute the sampling step in deg based on the wavelength of mode-1 waves
        delta_lon, delta_lat = get_delta(lat)
        #print('The distance per degree lon is %.1f km.' % delta_lon)
        #print('The distance per degree lat is %.1f km. \n' % delta_lat)
        step_lon, step_lat = get_radius_in_deg(kh,lon,lat,delta_lon,delta_lat)
        #print('The sampling steps are %.1f degree in lon and %.1f degree in lat.' % (step_lon, step_lat))
        step = max(step_lon, step_lat)
        if step < 1:
            step = 1
        if step > 3:
            step = 3

        # sample the topography
        topog_sample = depth.where((topog.lon>lon-step) & (topog.lon<lon+step) & (topog.lat>lat-step) & (topog.lat<lat+step), drop=True)
        #print('Mean of the sampled topog: \n', topog_sample.mean(skipna=True))
        
        
        # perform spectral analysis
        #print('Spectral analysis... \n')
        #topog_spd_1d, kh_1d, topog_spd_2d, kx_2d, ky_2d, dk, dl, dy, dx = spectral_analysis(topog_sample,lat,delta_lon,delta_lat)
        topog_spd_2d, dx, dy = fft_topog(topog_sample,delta_lon,delta_lat,k_grid_units=False)
        #print('Azimuthal summing... \n')
        #topog_spd_1d = azimuthal_sum(topog_spd_2d)
        var_int_x2 = np.nansum(topog_sample **2 * dy * dx)
        #print('2D sum, demeaned (x-space)  = %3.2e \n' % var_int_x2)
        #print('Mean of the sampled topog: \n', topog_sample.mean(skipna=True))

        # check Parseval's theorem
        print('Checking Parsevals theorem...')
        dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
        #dkh = np.max(np.diff(topog_spd_1d.k))
        fft2dsum = topog_spd_2d.sum().sum()*dkx*dky
        #fft1dsum = topog_spd_1d.sum()*dkh
        print('2D sum (k-space)     = %3.2e' % fft2dsum)
        #print('Radial sum (k-space) = %3.2e' % fft1dsum)
        var_int_x = np.nansum(topog_sample **2 * dy * dx)
        print('2D sum (x-space)     = %3.2e \n' % var_int_x)
        #print('Mean of the sampled topog: \n', topog_sample.mean(skipna=True).values)
        h_ms = np.nanmean(topog_sample ** 2)
        #print('mean-square height   = %3.2e \n' % h_ms)

        A_loc = 2 * step * delta_lon * 1e+3 * 2 * step * delta_lat * 1e+3 

        h_rms = compute_h_rms(A_loc,topog_spd_2d)
        print('rms height           = %3.2e \n' % h_rms)

        k_bar = compute_h_weighted_k(h_rms,A_loc,topog_spd_2d)
        L_bar = 2 * np.pi / k_bar / 1e+3
        print('height weighted mean k converted scale: %3.2e km \n' % L_bar)

        # compute the drag coefficient
        print('Computing the drag coeff...')
        sigma_xx, sigma_yy, sigma_xy = compute_drag_coeff(A_loc,N_loc,H_loc,omega,f_loc,topog_spd_2d)
        
    else:
        print('Outside critical latitude.')
        sigma_xx, sigma_yy, sigma_xy = 0, 0, 0

    print('Drag coefficient (x-dir): %3.2e' % sigma_xx)
    print('Drag coefficient (y-dir): %3.2e' % sigma_yy)
    print('Drag coefficient (cross): %3.2e' % sigma_xy)


    return sigma_xx, sigma_yy, sigma_xy

def get_N_clim(lon,lat):

    Nbot_data = xr.open_dataset('/g/data/nm03/lxy581/WOA18/Nbot_1000m_woa18.nc')

    # Read bottom N and convert the unit to rad/s
    Nbot_1km = Nbot_data.Nbot * 2 * np.pi

    # Use the 2D interpolation to find the bottom N
    N_clim = Nbot_1km.interp(lat=lat,lon=lon).values

    return N_clim

def estimate_kh(omega,f_loc,N_loc,H_loc):

    kh = np.sqrt((omega**2-f_loc**2)*np.pi**2/N_loc**2/H_loc**2)

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

def filtering(topog_sample):

    ydim, xdim = topog_sample.dims
    nx = topog_sample[xdim].size
    ny = topog_sample[ydim].size
    win_2d = np.full((ny,nx), np.nan)

    radi = topog_sample.shape[0] / 2
    print('nx, ny = ', nx, ny)
    print('topog_sample shape: ', topog_sample.shape)
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

def azimuthal_sum(topog_spd):

    # following DurranWeynMenchaca2017a http://dx.doi.org/10.1175/MWR-D-17-0056.1
    # omits zero wavenumber
    dkh = np.max([np.max(np.diff(topog_spd.kx)), np.max(np.diff(topog_spd.ky))])
    dkmin = np.min([np.min(np.diff(topog_spd.kx)), np.min(np.diff(topog_spd.ky))])

    # extends sqrt(2) times further to get into corners
    Nmax = int(np.ceil(np.sqrt(2)*max(topog_spd.shape)/2))
    kp = dkh*range(1,Nmax+1)

    # number of wavenumber points in each annulus (C in DurranWeynMenchaca2017a)
    C = 0.0*kp
    fftradial = 0.0*kp
    radius = np.sqrt(topog_spd.kx**2+topog_spd.ky**2)
    ones = 1 + 0*topog_spd

    # sum in each annulus
    for i,k in enumerate(kp):
        fftradial[i] = topog_spd.where(radius>=k-dkh/2).where(radius<k+dkh/2).sum()
        C[i] = ones.where(radius>k-dkh/2).where(radius<=k+dkh/2).sum()

    # scale as in eq (24) (assuming scaling in eq (22) is already done)
    fftradial *= dkmin

    # eq (26): compensate for number of (k,l) pairs in each annulus
    # Parseval's theorem no longer exactly holds (p 3905)
    C = np.where(C==0, 1, C)  # ensures no division by zero (fftradial=0 there anyway)
    fftradial *= 2.0*np.pi*kp/C/dkmin

    fftradial = xr.DataArray(fftradial, dims=['k'], coords={'k': kp})
    fftradial.k.attrs['units'] = topog_spd.kx.attrs['units']
    fftradial.k.attrs['long_name'] = 'wavenumber magnitude'

    # Truncate spectrum at Nyquist frequency (high k's in corners are anisotropically sampled):
    # Also breaks Parseval's theorem
    kminmax = np.min([np.max(topog_spd.kx), np.max(topog_spd.ky)])
    fftradial = fftradial.sel(k=slice(0.,kminmax))

    return fftradial

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

def compute_h_rms(A,topog_spd_2d):

    const = 1 / (4 * A * np.pi**2)
    kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
    dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
    int_kl = topog_spd_2d * dkx * dky
    h_rms = np.sqrt(const * np.nansum(int_kl[:]))

    return h_rms

def compute_h_weighted_k(h_rms,A,topog_spd_2d):

    const = 1 / (4 * A * np.pi**2)
    kx_2D, ky_2D = np.meshgrid(topog_spd_2d.kx,topog_spd_2d.ky,indexing='xy')
    dkx, dky = np.max(np.diff(topog_spd_2d.kx)), np.max(np.diff(topog_spd_2d.ky))
    k_mag = np.sqrt(kx_2D**2 + ky_2D**2)
    int_kl = k_mag * topog_spd_2d * dkx * dky
    k_bar = 1/ h_rms**2 * const * np.nansum(int_kl[:])

    return k_bar
