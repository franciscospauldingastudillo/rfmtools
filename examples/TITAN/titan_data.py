# Load dependencies
import xarray as xr
import climlab
import sys
sys.path.append('/home/fspauldinga/SAM24a/RFM/rfmtools')
import rfmtools
from scipy import ndimage
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure as mplf
import os
import time
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker,colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, ScalarFormatter, NullFormatter)
import matplotlib.pylab as pyl
from labellines import labelLines,labelLine
import pickle
from scipy.special import lambertw
from collections.abc import Sequence
from typing import Type,Dict
from scipy.interpolate import interp1d
if True:
    # Get plot settings
    pltfac = 2
    dim = 3.2*pltfac
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 10*pltfac}
    mpl.rc('font', **font)
    lticksmajor = 3*pltfac
    lticksminor = lticksmajor/2
    wticks = lticksmajor/4
    msize = 1.5*(4*pltfac)**2
    
#############################################################
def get_esat_over_l(par,T):
    import math
    # SVP over liquid from DAM (Pa)
    pstar = par.ptrip * (T/par.Ttrip)**((par.cpv-par.cvl)/par.rgasv) * math.exp( (par.E0v - (par.cvv-par.cvl)*par.Ttrip) / par.rgasv * (1/par.Ttrip - 1/T) )
    return pstar
#############################################################
    
def get_custom_atm(par,vres1=np.arange(0,5e4,1e2),vres2=np.arange(0,5e4,1e2)):
    # scalar inputs: mid-tropospheric RH (RHmid) and temeprature (Tmid) and boolean option for uniform RH
    # Vertical resolution can also be specified to facilitate RFM vs analytical model. 
    #############################################################
    def get_optimal_RH(Tsfc2trp, Ts=300, Tmid=260, Ttrp=200, RHs=0.75, RHmid=0.4, RHtrp=0.75):
        print('Ts',Ts)
        print('Tmid',Tmid)
        print('Ttrp',Ttrp)
        print('RHs',RHs)
        print('RHmid',RHmid)
        print('RHtrp',RHtrp)
        # Compute alpha_left and alpha_right based on target RH values
        alpha_left  = -np.log(RHtrp / RHmid)/(Ttrp - Tmid) ** 2
        alpha_right = -np.log(RHs / RHmid)/(Ts - Tmid) ** 2 
        # Compute RH values for the provided temperature range
        RH_opt = [
            RHmid * np.exp(-(temp - Tmid) ** 2 * alpha_left) if temp < Tmid else RHmid * np.exp(-(temp - Tmid) ** 2 * alpha_right)
            for temp in Tsfc2trp
        ]
        return {'RH':RH_opt,'alpha_lt':alpha_left, 'alpha_gt':alpha_right}
    #############################################################
    def get_esat_over_l(par,T):
        import math
        # SVP over liquid from DAM (Pa)
        pstar = par.ptrip * (T/par.Ttrip)**((par.cpv-par.cvl)/par.rgasv) * math.exp( (par.E0v - (par.cvv-par.cvl)*par.Ttrip) / par.rgasv * (1/par.Ttrip - 1/T) )
        return pstar
    #############################################################
    # Lapse Rate and Temperature (Troposphere)
    Gamma = np.ones([len(par.z)],dtype='float')*par.Gamma
    T  = par.Ts - par.Gamma*par.z
    # stratospheric mask
    mask = np.where(T<par.Ttrp)
    # Lapse Rate and Temperature (Stratosphere)
    T  = np.where(T<par.Ttrp,par.Ttrp,T)
    Gamma[mask] = 0
    # Identify the height of the tropopause
    ktrp = int(np.amin(np.where(T==par.Ttrp)))
    ztrp = par.z[ktrp]
    # Custom RH
    if par.uniform is True:
        RH       = np.ones([len(par.z)])*par.RHmid
        RH[mask] = 0
        alpha_lt = 0
        alpha_gt = 0
    else:
        RH  = np.ones([len(par.z)])*par.RHs
        RH[mask] = 0 # stratospheric mask
        foo = get_optimal_RH(T[0:(ktrp+1)], par.Ts, par.Tmid, par.Ttrp, par.RHs, par.RHmid, par.RHtrp)
        RH[0:(ktrp+1)] = foo['RH']
        alpha_lt       = foo['alpha_lt']
        alpha_gt       = foo['alpha_gt']
    
    # Solve for environmental pressure and density 
    rho  = np.zeros_like(par.z)
    p    = np.ones_like(par.z)*par.ps
    arg  = 0
    # initialize molar mixing ratios (relative to total)
    xN2  = np.zeros_like(par.z)
    xCH4 = np.zeros_like(par.z)
    xH2  = np.ones_like(par.z)*par.xH2
    # initialize mass mixing ratios (relative to total)
    wN2  = np.zeros_like(par.z)
    wCH4 = np.zeros_like(par.z)
    wH2  = np.zeros_like(par.z)
    # initialize specific gas constant
    Rtot  = np.zeros_like(par.z)
    # initialize mean molecular mass
    Mave  = np.zeros_like(par.z)
    for k in range(len(par.z)):
        if k<(len(par.z)-1):
            dz   = par.z[k+1]-par.z[k]
        else:
            dz   = par.z[1]-par.z[0]
        # molar mixing ratio of CH4 
        if k<=ktrp: # tropospheric value set by Clausius-Clapeyron
            pCH4    = RH[k]*get_esat_over_l(par,T[k])
            xCH4[k] = pCH4/p[k]
        else: # stratospheric mixing ratio fixed to tropopause value
            xCH4[k] = xCH4[ktrp]
        # molar mixing ratio of N2
        xN2[k]  = 1-xCH4[k]-xH2[k]
        # mean molecular weight of air
        Mave[k] = (1-xCH4[k])*par.MN2 + xCH4[k]*par.MCH4 + xH2[k]*(par.MH2-par.MN2)
        # mass mixing ratios of H2, CH4, and N2
        wH2[k] = xH2[k]*par.MH2/Mave[k]
        wN2[k] = xN2[k]*par.MN2/Mave[k]
        wCH4[k]= xCH4[k]*par.MCH4/Mave[k]
        tol = 1e-6  # Adjust this value based on your precision requirements
        if abs(wH2[k] + wN2[k] + wCH4[k] - 1) > tol:
            print(f'Error: sum of mixing ratios is non-unity ({wH2[k] + wN2[k] + wCH4[k]}). Tolerance exceeded.')
        # specific gas constant of air
        Rtot[k] = wN2[k]*par.RN2 + wCH4[k]*par.RCH4 + wH2[k]*par.RH2
        # solve for total density of air
        rho[k] = p[k]/(Rtot[k]*T[k])
        # solve for exponential term
        arg    = -par.ggr/Rtot[k]*dz/T[k]
        # solve for pressure at next level
        if k<(len(par.z)-1):
            p[k+1] = p[k]*np.exp(arg)
    #############################################################
    # Export fields in their desired vertical resolution (vres1,vres2)
    def interpolate(var, vres):
        # func = interp1d(par.z,var,kind)(input of function)
        return interp1d(par.z, var, kind='cubic')(vres)
    T1, T2         = interpolate(T, vres1), interpolate(T, vres2)
    Gamma1, Gamma2 = interpolate(Gamma, vres1), interpolate(Gamma, vres2)
    p1, p2         = interpolate(p, vres1), interpolate(p, vres2)
    rho1, rho2     = interpolate(rho, vres1), interpolate(rho, vres2)
    RH1, RH2       = interpolate(RH, vres1), interpolate(RH, vres2)
    xN2_1, xN2_2   = interpolate(xN2, vres1), interpolate(xN2, vres2)
    xCH4_1, xCH4_2 = interpolate(xCH4, vres1), interpolate(xCH4, vres2)
    xH2_1, xH2_2   = interpolate(xH2, vres1), interpolate(xH2, vres2)
    #############################################################
    dat1 = {'T': T1, 'p': p1, 'Gamma': Gamma1, 'rho': rho1, 'z': vres1, 'RH': RH1,
            'xN2': xN2_1, 'xCH4': xCH4_1, 'xH2': xH2_1}
    dat2 = {'T': T2, 'p': p2, 'Gamma': Gamma2, 'rho': rho2, 'z': vres2, 'RH': RH2,
            'xN2': xN2_2, 'xCH4': xCH4_2, 'xH2': xH2_2}
    #############################################################
    # T(K), RH(unitless), p(Pa), Gamma(K/m), rho(kg/m3), x(molar mixing ratio)
    #############################################################
    return dat1,dat2
   
def get_hr(dataset):
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    heights = dataset['RFM']['z']
    #############################################################
    srhr = [] # spectrally-resolved heating rate at each height (levs,nus)
    hr   = [] # spectrally-integ.   heating rate at each height (levs)
    fdir = '/home/fspauldinga/SAM24a/RFM/outp/%s'%par.case
    for height in heights:
        if height<100:
            df = '%s/coo_0000%i.asc' %(fdir,height)
        elif height<1000:
            df = '%s/coo_00%i.asc' %(fdir,height)
        elif height<10000:
            df = '%s/coo_0%i.asc' %(fdir,height)
        else:
            df = '%s/coo_%i.asc' %(fdir,height)
        # append spect-res. hr at each height
        srhr.append(np.loadtxt(df,skiprows=4))   
    srhr = np.array(srhr)    
    
    # Retain only the band information
    srhr = srhr[:,par.i0:par.i1+1] 
    
    # Calculate the band-integrated cooling rate (K/day)
    hr.append(par.dnu*np.sum(srhr,axis=1)) 
    hr = np.squeeze(np.array(hr))
    return {'hr':hr,'srhr':srhr} 


def run_RFM(dataset):
    # The inputs must be provided at the desired RFM resolution.
    #############################################################
    par     = dataset['par']
    runtype = par.runtype
    #############################################################
    TEMREL = par.TEMREL # sea-air dT (>0 means warmer surface))
    heights = dataset['RFM']['z']               # Height (m)
    nus = par.nus                               # Wavenumbers (cm-1)
    temps = dataset['RFM']['T']                 # Temperature (K)
    p = dataset['RFM']['p'] / 1e2               # Pressure (hPa)
    #############################################################
    # Tropopause information
    ktrp = np.amin(np.where(temps == par.Ttrp))  # Tropopause index
    ztrp = heights[ktrp]                         # Tropopause height
    #############################################################
    # Automatically handle all gas species
    print("run_RFM: Including the following gases:")
    xgases = {}
    for key, value in dataset['RFM'].items():
        if key.startswith('x'):  # Identify gas mixing ratios (e.g., xN2, xCH4)
            xgases[key] = value * 1.0e6  # Convert from mol/mol->ppmv
            print(f"  - {key[1:].upper()} [ppmv]")
    #############################################################
    # Generate the RFM input file dynamically
    rfmtools.make_input_files.generate_atm_file(
        f'{par.case}.atm',
        heights / 1e3,  # Convert to km
        temps,
        p,
        **xgases  # Pass all detected gases' molar mixing ratios [ppmv]
    )
    #############################################################
    # Create LEV file (space-separated vertical coordinates in km)
    np.savetxt(rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(par.case), heights/1e3, delimiter=' ')
    #############################################################
    # Generate RFM driver
    # https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
    PHY = (' CPKMOL=29012.0' + '\n' +        # molar heat capacity of air (J/K/kmol) (used for COO only)
                 ' GRAVTY=1.35'    + '\n' +  # m/s^2 (ignored by HYD flag)
                 ' RADCRV=2575.'   + '\n' +  # local radius of curvature (km)
                 ' TEMSPA=2.7'     + '\n' +  # cosmic background temperature (K)
                 ' WGTAIR=100000.')          # molar mass of air (kg/kmol) (ignored by HYD flag)
    print(PHY)
    # Extract active gases from the GAS list
    gases = [key[1:].upper() for key in xgases]  # e.g., ['N2', 'CH4', 'H2']
    # Optional collision-induced absorption (CIA) files
    ciafiles = [
        f"{rfmtools.utils.RFM_DIR}/hit/N2-N2_0_5000.cia",
        f"{rfmtools.utils.RFM_DIR}/hit/N2-H2_0_1886.cia",
        f"{rfmtools.utils.RFM_DIR}/hit/N2-CH4_0_1379.cia",
        f"{rfmtools.utils.RFM_DIR}/hit/CH4-CH4_200_800.cia",
    ]
    # Filter CIA files to only include valid pairs in par.valid_ciapairs that both belong to 'gases'
    filtered_ciafiles = []
    for ciafile in ciafiles:
        filename = ciafile.split('/')[-1]  # Get the filename (e.g., 'N2-H2_0_1886.cia')
        pair = filename.split('_')[0]      # Extract molecule pair (e.g., 'N2-H2')
        mol1, mol2 = pair.split('-')       # Split into two molecules
        # Check if this CIA pair is valid AND both molecules are in 'gases'
        if ((mol1, mol2) in par.valid_ciapairs or (mol2, mol1) in par.valid_ciapairs) and (mol1 in gases and mol2 in gases):
            filtered_ciafiles.append(ciafile)
    # Generate shared arguments for the RFM drivers
    common_args = {
        'runtype': par.runtype,
        'fname': 'rfm.drv',
        'atmfile': f'{par.case}.atm',
        'SPC': f"{par.nu0} {par.nu1} {par.dnu}",
        'GAS': " ".join(gases),
        'HIT': f"{rfmtools.utils.RFM_DIR}/hit/titan_n2_ch4_h2_1_3000_hitran20.par",
        'OUTDIR': f"{rfmtools.utils.RFM_DIR}/outp/{par.case}",
        'PHY': PHY,
        'NLEV': f"{rfmtools.utils.RFM_DIR}/lev/{par.case}.lev",
        'CIA': "\n".join(filtered_ciafiles)  # Include only matching CIA files
    }
    print('Assembling RFM driver:',common_args)
    # Optional: print the included CIA files for verification
    print("Included CIA files:")
    if filtered_ciafiles:
        for cia in filtered_ciafiles:
            print(f"  - {cia}")
    else:
        print("  - None (No valid CIA pairs found)")
    # runtype-specific configurations (delayed functional mappings)
    runtype_mapping = {
        'cooling': lambda: rfmtools.make_input_files.make_driver(**common_args, SFC=f"TEMREL={TEMREL:.1f}"),
        'continuum_cooling': lambda: rfmtools.make_input_files.make_driver(**common_args, SFC=f"TEMREL={TEMREL:.1f}"),
        'od_trans': lambda: rfmtools.make_input_files.make_driver(**common_args),
        'continuum_od_trans': lambda: rfmtools.make_input_files.make_driver(**common_args),
        'kabs': lambda: rfmtools.make_input_files.make_kabs_driver(**common_args, TAN="1 550 1 82 1 100"),
        'continuum_kabs': lambda: rfmtools.make_input_files.make_kabs_driver(**common_args, TAN="1 550 1 82 1 100"),
    }
    # run the appropriate driver
    if runtype in runtype_mapping:
        runtype_mapping[runtype]()
    else:
        raise ValueError(f"Unsupported runtype: {runtype}")
    #############################################################
    # Run RFM    
    print(rfmtools.utils.RFM_DIR)
    rfmtools.run.run(drv_file=f"{rfmtools.utils.RFM_DIR}/src/rfm.drv")
    #############################################################
    print(f'done with RFM for '+ par.case)
    return
            
def get_odepth(dataset)->dict[str,float]:
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    heights = dataset['RFM']['z']
    #############################################################
    # odepth uses an internal decision for crdnu
    crdnu = 40 # cm-1
    #############################################################
    srtau    = []  # spectrally-resolved optical depth at each height (levs,nus)
    zsrtau1  = []  # tau=1 heights as a function of nu (nus)
    #D        = 1.5 # two-stream diffusivity factor
    fdir     = rfmtools.utils.RFM_DIR+'/outp/%s'%par.case
    for height in heights:
        if height<100:
            df = '%s/opt_0000%i.asc' %(fdir,height)
        elif height<1000:
            df = '%s/opt_00%i.asc' %(fdir,height)
        elif height<10000:
            df = '%s/opt_0%i.asc' %(fdir,height)
        else:
            df = '%s/opt_%i.asc' %(fdir,height)
        # append spect-res. tau at each height
        srtau.append(np.loadtxt(df,skiprows=4))   
    #srtau = D*np.array(srtau)
    srtau  = np.array(srtau)
      
    # "walk the line" and find z(nu | tau=1)
    for n,nu in enumerate(par.nus):
        #k1 = np.where(np.abs(1-D*srtau[:,n])==np.min(np.abs(1-D*srtau[:,n])))
        k1 = np.where(np.abs(1-srtau[:,n])==np.min(np.abs(1-srtau[:,n])))
        zsrtau1.append(heights[k1])
    zsrtau1 = np.squeeze(np.array(zsrtau1))
    
    # retain information from the spectral band
    srtau   = srtau[:,par.i0:par.i1+1]
    zsrtau1 = zsrtau1[par.i0:par.i1+1]
        
    # Compute linear averages over coarse 10 cm-1 bins
    crnus     = np.arange(par.nu0+crdnu/2,par.nu1+crdnu/2,crdnu)
    bin_edges = np.arange(par.nu0,par.nu1+crdnu,crdnu)
    bin_means,bin_edges,binnumber = scipy.stats.binned_statistic(par.nus[par.i0:par.i1+1], srtau, statistic='mean', bins=bin_edges)
    crsrtau   = bin_means
    
    bin_edges = np.arange(par.nu0,par.nu1+crdnu,crdnu)
    bin_means,bin_edges,binnumber = scipy.stats.binned_statistic(par.nus[par.i0:par.i1+1], zsrtau1, statistic='mean', bins=bin_edges)
    crzsrtau1 = bin_means
        
    return {'srtau':srtau,'zsrtau1':zsrtau1,'crnus':crnus,'crsrtau':crsrtau,'crzsrtau1':crzsrtau1}

def get_bint_planck(dataset)->Dict[str,float]:
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    heights = dataset['RFM']['z']
    temps   = dataset['RFM']['T']
    #############################################################
    def get_B(nu:Sequence[float],T:int)->Dict[str,float]:
        # Planck emission ((W/m2)*cm) as a function of wavenumber (nu; array; cm-1) and temperature (T; scalar; K)
        nu = nu*100 # 1/cm -> 1/m
        h = 6.626e-34 # J*s
        c = 3.0e8     # m/s
        k = 1.38e-23  # J/K
        B = 2*h*c**2*nu**3/(np.exp((h*c*nu)/(k*T))-1) # kg*m/s^2 = W m^(-2) m^(+1)
        B = 100*B # W m^(-2) m^(+1) -> W*cm/m^2
        return {'B':B}
    #############################################################
    # Compute the band-integrated planck emission
    #############################################################
    piBsr    = [] # spectrally-resolved planck emission (W*cm/m^2)
    piBbar   = [] # band-integrated planck emission (W/m^2)
    #############################################################
    # at each height, get spectrally-resolved B and band-integrate
    for k,height in enumerate(heights):
        data = get_B(par.nus[par.i0:par.i1+1],temps[k])
        integrand = np.pi*data['B']
        piBsr.append(integrand)
        piBbar.append(np.trapz(integrand,par.nus[par.i0:par.i1+1]))
    piBsr    = np.array(piBsr)
    piBbar = np.array(piBbar)
    return {'piBsr':piBsr,'piBbar':piBbar}   

def get_trans(dataset)->Dict[str,float]:
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    heights = dataset['RFM']['z']
    p       = dataset['RFM']['p']
    #############################################################
    Trsr    = []  # spectrally-resolved tranmissivity [Ttv = exp(-ttau)] at each height (z,nus)
    Tr      = []  # band-integrated transmissivity at each height (z)
    dpTrsr  = []  # spectrally-resolved transmissivity gradient at each height (zint,nus)
    dpTr    = []  # band-integrated transmissivity gradient at each height (zint)
    dzTrsr  = []  # spectrally-resolved transmissivity gradient at each height (zint,nus)
    dzTr    = []  # band-integrated transmissivity gradient at each height (zint)
    eemi    = []  # effective emissivity at each height (zint)
    emisr   = []  # spectrally-resolved emissivity at each height (zint)
        
    # Calculate the spectrally-resolved transmissivity (z,nus)
    fdir = rfmtools.utils.RFM_DIR+'/outp/%s'%par.case
    for height in heights:
        if height<100:
            df = '%s/tra_0000%i.asc' %(fdir,height)
        elif height<1000:
            df = '%s/tra_00%i.asc' %(fdir,height)
        elif height<10000:
            df = '%s/tra_0%i.asc' %(fdir,height)
        else:
            df = '%s/tra_%i.asc' %(fdir,height)
        # append spect-res. Tr at each height
        Trsr.append(np.loadtxt(df,skiprows=4))    
    Trsr = np.array(Trsr)
    Trsr = Trsr[:,par.i0:par.i1+1]
    
    # Calculate the band-integrated transmissivity (z)
    Tr.append(par.dnu*np.sum(Trsr,axis=1))
    Tr = np.squeeze(np.array(Tr))
        
    # Calculate the spectrally-resolved transmissivity gradient (zint,nus) over band
    for k in range(len(heights)-1):
        srval = (Trsr[k+1,:]-Trsr[k,:])/(p[k+1]-p[k])
        dpTrsr.append(srval)
    dpTrsr = np.array(dpTrsr)
        
    # Calculate the band-integrated transmissivity gradient (zint)
    for k in range(len(heights)-1):
        val = (Tr[k+1]-Tr[k])/(p[k+1]-p[k])
        dpTr.append(val)
    dpTr = np.array(dpTr)
    
    # Calculate the spectrally-resolved transmissivity gradient (zint,nus) over band
    for k in range(len(heights)-1):
        srval = (Trsr[k+1,:]-Trsr[k,:])/(heights[k+1]-heights[k])
        dzTrsr.append(srval)
    dzTrsr = np.array(dzTrsr)
        
    # Calculate the band-integrated transmissivity gradient (zint)
    for k in range(len(heights)-1):
        val = (Tr[k+1]-Tr[k])/(heights[k+1]-heights[k])
        dzTr.append(val)
    dzTr = np.array(dzTr)