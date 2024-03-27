import time
from pylab import *
import numpy as np
import xarray as xr
from math import atan2

def fitSine(tList,yList,freq=1/12.42):
    '''
       freq in cycle per hour
       tList in hours
    returns
       phase in radians
    '''
    b = matrix(yList).T
    rows = [ [cos(freq*2*pi*t), sin(freq*2*pi*t), 1] for t in tList]
    A = matrix(rows)
    (w,residuals,rank,sing_vals) = lstsq(A,b,rcond=None)
    phase = atan2(w[1,0],w[0,0])
    amplitude = norm([w[0,0],w[1,0]],2)
    bias = w[2,0]
    return (phase,amplitude,bias)

def get_amp_phase(elev_xr,xind,yind,time_len):
    time = np.arange(time_len)
    elev = np.array(elev_xr.isel({'xh':xind,'yh':yind,'zi':0,'time':np.arange(time_len)}))
    (phaseEst,amplitudeEst,biasEst) = fitSine(time,elev,freq=1/12.42)
    return amplitudeEst,phaseEst/np.pi*180

file = '/g/data/nm03/lxy581/archive/tides_025_JSL/output011/ocean_interior.nc'
data = xr.open_dataset(file)
elev_xr  = data.e
time_len = 24*7
nx, ny   = data.sizes['xh'], data.sizes['yh']
amp = np.full((ny,nx),np.nan)
pha = np.full((ny,nx),np.nan)
start_time = time.time()
for j in range(ny):
    print(j)
    for i in range(nx):
        amp[j,i], pha[j,i] = get_amp_phase(elev_xr,i,j,time_len)
end_time = time.time()
exe_time = float(end_time - start_time)
print("Execution time: %.1f seconds! \n" % exe_time)

est_data = xr.Dataset(data_vars={"amp": (["ny","nx"], amp),
                                 "phase": (["ny","nx"], pha)},
                      coords={"yh": (["ny"], np.array(data.yh)),
                              "xh": (["nx"], np.array(data.xh))})
est_data.to_netcdf('/g/data/nm03/lxy581/evaluate/amp_phase/tides_025_JSL_global.nc')

