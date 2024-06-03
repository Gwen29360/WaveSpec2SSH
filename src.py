import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt


from scipy.interpolate import griddata

import time
import xarray as xr

from src import *
g = 9.81 #m/sec^2

def invert_ef_spec_2_ssh2D(spec, s, dx, N, nth, theta_m):
    """
    Purpose: inverse a one-dimensional frequency spectrum into a realistic 2D Sea Surface Elevation field (wavy surface).
    ---------
    Inputs:
    ---------
    spec: the wave spectrum DataArray (Xarray) (works well with .nc file provided by CDIP).
    Your variables names have to be consistent with CDIP's variable name. (https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/029p1/029p1_historic.nc.html)
    s: the directional parameter (has to be integer)
    dx: the pixel size both in x and y for the inveted SSH field. Make it sufficiently small to resolve the highest frequencies
    N: the number of point for the square domain in meter
    nth: the number of directions (has to be >24)
    theta_m: the dominant direction of the wave in radian (theta_m = 0, wave are propagating from the left to the right)
    Outputs:
    ---------
    ds_ssh: the DataArray that contains the x-y axes and the elevation field
    spec_2d: the created two-dimensional wave spectrum + Qp, Hs, Lambda_p
    
    """
    
    # Make some tests
    assert isinstance(s, int), "s has to be an integer" # Verify that s is an integer
    assert nth>=24, "Need more direction (>24)"
    assert dx<=4, "pixel size too big"
    assert isinstance(spec, xr.Dataset), "Input data must be an xarray.DataArray"
        
    # Prepare the axes
    df = np.gradient(spec.waveFrequency.values)
    theta = np.linspace(0, 2*np.pi, nth)
    
    # Creation of the directional spectrum
    S = (np.cos((theta - theta_m)/2))**(2*s) # The Directional distribution
    S_2d = np.matlib.repmat(S, len(spec.waveFrequency.values), 1)
    psd = spec.waveEnergyDensity.values

    dirSpec = np.array(psd) * S_2d.T  # E(f, th) = E(f) * D(th)
    # load in xarray dataset
    K = ((2*np.pi*spec.waveFrequency.values)**2)/g # uses the deep water disp. rel. to go from freq. to wavenumber
    KK, DD = np.meshgrid(K, theta)
    
    # Prepare the wavenumber
    Kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, dx))
    Ky = Kx
    ky_grid, kx_grid = np.meshgrid(Ky, Kx)
    k_grid = (kx_grid**2 + ky_grid**2)**.5
    theta_grid = np.arctan2(ky_grid, kx_grid)
    theta_grid[theta_grid<0] = theta_grid[theta_grid<0]+2*np.pi
    
    # interpolation to get E_kxky
    points = (DD.flatten(), KK.flatten()) # Original coordinates
    values = dirSpec.flatten() # Data in the original coordinates
    xi = (theta_grid.flatten(), k_grid.flatten()) # Target grid points
    cart_grid = griddata(points, values, xi, method = 'linear')
    Ekxky = cart_grid.reshape(k_grid.shape)
    
    
    Ekxky[np.isnan(Ekxky)] = 0 # Zeroing the amplitude at wavenumbers higher than k_max
    pha = 2*np.pi*(np.random.rand(*Ekxky.shape) - 1)

    F_hat = (np.cos(pha) + 1j*np.sin(pha))*Ekxky**.5 # assings random phase
    F_hat = np.fft.fftshift(F_hat)
    
    # Inverse E_kxky to get ssh
    eta = np.fft.ifft2(F_hat, axes=(0,1))
    eta = eta.real

    # Normalize to desired significant wave height
    hs = spec.waveHs.values
    eta_norm = hs*eta.real/eta.std()/4

    
    # Save spectral parametrization
    spec_2d = dirSpec
    fp = spec.waveFrequency[spec.waveEnergyDensity.argmax()].values
    Lp = 2*np.pi/((2*np.pi*fp)**2/(g))
    Qp = (np.sum(spec.waveEnergyDensity.values*df))**2/(np.sum(spec.waveEnergyDensity.values**2*df))
    
    # Save in Xarray
    x = np.arange(0, N*dx, dx)
    
    ssh = xr.DataArray(data=eta_norm, dims=['y', 'x'],
                  coords=dict(x=x, y=x))
    
    
    dirSpec = xr.DataArray(data=Ekxky, dims=['kx', 'ky'],
            coords=dict(kx=Kx, ky=Ky))
    
    #print(dirSpec)
    spec_2d = xr.Dataset()
    spec_2d['kxky_spectrum'] = dirSpec
    spec_2d.kx.attrs['units'] = 'rad/m'
    spec_2d.kx.attrs['long_name'] = 'zonal wavenumber'
    spec_2d.ky.attrs['units'] = 'rad/m'
    spec_2d.ky.attrs['long_name'] = 'meridional wavenumber'
    spec_2d.attrs['significant_wave_height'] = '{:.3f} m'.format(hs)
    spec_2d.attrs['peak_wavelength'] = '{:.3f} m'.format(Lp)
    spec_2d.attrs['peakedness_param'] = '{:.3f}'.format(Qp)
    spec_2d.attrs['creator'] = 'Gwendal Marechal'
    spec_2d.attrs['data_created'] = time.ctime()
    
    ds_ssh = xr.Dataset()
    ds_ssh['sea_surface_height'] = ssh
    ds_ssh.sea_surface_height.attrs['units'] = 'm'
    ds_ssh.x.attrs['units'] = 'm'
    ds_ssh.x.attrs['long_name'] = 'distance along track'
    ds_ssh.y.attrs['units'] = 'm'
    ds_ssh.y.attrs['long_name'] = 'distance across track'
    ds_ssh.sea_surface_height.attrs['long_name'] = 'sea surface height'
    ds_ssh.attrs['creator'] = 'Gwendal Marechal'
    ds_ssh.attrs['data_created'] = time.ctime()

    return ds_ssh, spec_2d



def ssh_1D_time_from_kspec(n_point, psd_k, wavenumber, hs):
    """
    purpose: estimate the one dimensional surface elevation space serie from one dimensional wavenumber spectrum
    ---------
    
    inputs:
    ---------
    n_point: length of the time serie in sec
    psd_k: the one dimensional wavenumber spectrum
    wavenumber: the wavenumber axis [rad/m]
    frequencies: the frequency axis [Hz]
    hs: the measured significant wave height
    
    outputs:
    ---------
    x: the space axis
    zeta_1d: the wave elevation reconstructed
    """
    
    wavenumber = np.array(wavenumber)
    
    N = n_point # length of the time series (in sec)
    
    #--- frequency, wavenumber and space steps
    #DF = np.amax(frequency)
    DK = np.amax(wavenumber)
    DX = (2 * np.pi) / DK
    dk0 = np.gradient(wavenumber)
    x_axis = np.squeeze(np.linspace(0, DX * 2* N, 2*N))
    dx = np.diff(x_axis)[0] # space resolution

    f = interpolate.interp1d(wavenumber, psd_k)
    k_new = np.linspace(np.amin(wavenumber), np.amax(wavenumber), N)
    
    #--- use interpolation function returned by `interp1d`
    dk = np.gradient(wavenumber)
    psd_new = f(k_new)
    
    #--- Inverse spectrum
    magnitude = np.sqrt(psd_new)

    #--- random phase
    phase = 2 * np.pi * np.random.randn(N)
    #zeta1d = np.zeros((n_point))
    FFT = magnitude * phase
    zeta1d = np.real(np.fft.ifft(FFT))
    eta_norm = hs * zeta1d / np.std(zeta1d) / 4

    x = x_axis[0:len(eta_norm)//2]
    zeta_1d = eta_norm[0:len(eta_norm)//2]
    return x, zeta_1d



def ssh_1D_time_from_fspec(n_point, psd, frequencies, hs):
    """
    purpose: estimate the one dimensional surface elevation time serie from one dimensional frequency spectrum
    ---------
    inputs:
    ---------
    n_point: length of the time serie in sec
    psd: the one dimensional frequency spectrum
    frequencies: the frequency axis [Hz]
    hs: the measured hs
    
    outputs:
    ---------
    t: the time axis
    zeta_1d: the wave elevation reconstructed
    """
    N = n_point # length of the time series (in sec)
    fp = frequencies[np.where(psd == np.amax(psd))]
    fmax = np.amax(frequencies)
    df = np.mean(abs(np.gradient(frequencies)))
    
    dt = 1/(2*df) # The temporal resolution
    f = interpolate.interp1d(frequencies, psd)
    
    freq_new = np.linspace(np.amin(frequencies), np.amax(frequencies), N)

    psd_new = f(freq_new) # use interpolation function returned by `interp1d`
    df = np.gradient(freq_new)

    magnitude = np.sqrt(psd_new) / 2
    t = np.linspace(0, N, N)

    #--- random phase
    phase = 2 * np.pi * np.random.randn(N)
    FFT = magnitude * np.exp(1j * phase)

    zeta1d = np.fft.ifft(FFT)
    
    zeta_1d = hs * zeta1d / np.std(zeta1d) / 4
   #print(np.shape(zeta1d))

    return t, zeta_1d

def E_kxky_to_kth(Ekxky, kx, ky, nd):
    
    """
    Purpose:
    ---------
    Switch from the kx, ky wave spectrum into the wavenumber-direction with the Jacobian K = (kx**2 + ky**2)**(1/2)
    Inputs:
    ---------
    Ekxky: the kx,ky wave spectrum
    kx,ky: the along across track wavenumbers respectively
    nd: number of direction considered
    
    Outputs: The datarray with the omnidirectional and the directional wavenumber spectra
    ---------
    """
    ##########
    # --- Initialize the axes
    ##########
    kkx, kky = np.meshgrid(kx, ky)
    kk = (kkx**2 + kky**2)**.5
    dd = np.arctan2(kky, kkx)
    nk = np.amax([len(kx), len(ky)])
    
    #k_new = np.linspace(0.002, np.amax(ky), nk)
    k_new = np.linspace(2*np.pi/600, 2*np.pi/20, nk)
    d_new = np.linspace(-np.pi, np.pi, nd)
    kk_new, dd_new = np.meshgrid(k_new, d_new)
    
    theta = d_new + np.pi/2
    dth = theta[1] - theta[0]
    
    ##########
    # --- Initialize the datarray
    ##########
    
    ds = xr.Dataset({
        'kx': ('kx', kx),
        'ky': ('ky', ky),
        'Ekxky': (['kx', 'ky'], Ekxky)
    })
    
    ds = ds.set_coords(('kx', 'ky'))
    ##########
    # --- Loop over direction to get kx and ky for the k interpolation
    ##########

    E = ds.Ekxky.isel()
    dirspec = xr.DataArray(np.zeros(dd_new.shape) * np.nan, [('theta', theta),  ('K', k_new)])
    for i in range(nd):
            di = dd_new[i][0]
            ki = k_new
            kx_int = xr.DataArray(ki*np.cos(di), dims='K')
            ky_int = xr.DataArray(ki*np.sin(di), dims='K')
            dirspec[i] = E.interp(ky=ky_int, kx=kx_int)
    ds_out = xr.Dataset({'K': ('K', k_new),
                    'theta': ('theta', theta),
                    })
    ds_out['Ekth'] = dirspec*kk_new  # Apply the Jacobian
    ds_out['Ek'] = ds_out['Ekth'].sum(dim='theta')*dth

    ds_out = ds_out.set_coords(('K', 'theta'))

    return ds_out
    

def E_kth_to_Efth(spec):
    """
    Purpose:
    ________
    Switch from the k, theta wave spectrum into the frequency-direction
    Inputs:
    The datarray obtained with the function E_kxky_to_kth
    
    Outputs: The datarray with the omnidirectional and the directional frequency spectra
    @ g.marechal September 2023
    """
    assert isinstance(spec, xr.Dataset), "Input data must be an xarray.DataArray"

    g = 9.81 # m/s^2
    f = (g*spec.K.values)**(1/2)/(2*np.pi)  # The frequency from the deep water disp relationship
    cg = 1/2 * (g/spec.K.values)**(1/2)  # The group velocity from the deep water disp relationship
    

    dirspec_f = spec.Ekth.values * (2 * np.pi)/cg
    dirspec_f = xr.DataArray(dirspec_f, [('theta', spec.theta.values),  ('F', f)])
    
    ds_out_bis = xr.Dataset({'frequency': ('F', f),
                    'theta': ('theta', spec.theta.values),
                    })
    
    ds_out_bis['Efth'] = dirspec_f
    dth = np.diff(spec.theta.values)[0]
    ds_out_bis['Ef'] = ds_out_bis['Efth'].sum(dim='theta')*dth
    ds_out_bis = ds_out_bis.set_coords(('frequency', 'theta'))
    
    return ds_out_bis




def plot_wave_spectrum(spec, vmin, vmax, date_obs, date_on):
    """
    Purpose:
    ---------
    Plot the frequency-direction wave spectrum
    
    Inputs:
    ---------
    spec: the frequency-direction spectrum Datarray
    vmin: the min value plot
    vmax: the max value plot
    date_obs: the date of the observations
    date_on: do you want to plot the date of obs? (yes or something else)
    
    Outputs: empty
    ---------
    """
    assert isinstance(spec, xr.Dataset), "Input data must be an xarray.DataArray"
    props = dict(boxstyle='round', facecolor='w', alpha=1)



    DIR0, FREQ0 = np.meshgrid(spec.theta.values, spec.frequency.values)
    
    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize = (2.5, 2.5))

    p1 = plt.contourf(DIR0, FREQ0, spec.Efth.values.T, np.linspace(vmin, vmax, 30), extend = 'both', cmap = 'plasma')
    plt.ylim([0,0.3])
    ax.set_rmax(.20)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.tight_layout()
    angle_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W','NW']
    ax.set_rlabel_position(180 + 45)
    #ax.set_ylim(0, 30)#set_rmax(30)
    #ax.set_ylabels('Wave period [s]', labelpad = 20)
    ax.set_thetagrids(angles = range(0, 360, 45),
                          labels = angle_labels,fontsize=12)
    ax.set_ylim([.01, .18])

    #############,
    #---White circles (based on the dispersion relationship in deep water)
    #############
    radius_050m=np.sqrt(g*2*np.pi/(50.*4*np.pi**2))
    radius_100m=np.sqrt(g*2*np.pi/(100.*4*np.pi**2))
    radius_200m=np.sqrt(g*2*np.pi/(200.*4*np.pi**2))
    radius_500m=np.sqrt(g*2*np.pi/(500.*4*np.pi**2))


    circle2 = plt.Circle((0.0, 0.0), radius_100m, color='white',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle2)
    circle3 = plt.Circle((0.0, 0.0), radius_200m, color='w',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle3)
    circle4 = plt.Circle((0.0, 0.0), radius_500m, color='w',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle4)
    #plt.text(0,radius_050m,'50 m',color='w')
    plt.text(0, radius_100m+.02,'100 m',color='w', fontsize = 9)
    plt.text(0, radius_200m+.02,'200 m',color='w', fontsize = 9)
    plt.text(0, radius_500m+.02,'500 m',color='w', fontsize = 9)

    ax.set_yticklabels([])
    if date_on=='yes':
        ax.set_title('%s'%str(date_obs))

    ax.grid(False)
    cbar_ax = fig.add_axes([0.05, .01, .3, 0.05])
    cbar=fig.colorbar(p1, cax=cbar_ax, shrink=0.5,aspect=155,extend='max',orientation='horizontal', ticks = [vmin, (vmin+vmax)/2, vmax])
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel(' [m²/rad/Hz]',rotation=0)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
