import numpy as np
from typing import Type,Dict,Tuple
import scipy
from collections.abc import Sequence

RFM_DIR = "/home/fspauldinga/SAM24a/RFM"

def read_spec ( filename ):
    with open(filename) as f:
        rec = '!'
        while rec[0] == '!': rec = f.readline()
        flds = rec.split()
        npts = abs ( int(flds[0]) )
        wno1 = float(flds[1])
        wnod = float(flds[2])
        if wnod > 0:         # regular grid
            spc = np.fromfile(f,sep=" ")
            wno = wno1 + np.arange(npts)*wnod
        else:                # irregular grid
            dat = np.fromfile(f,sep=" ").reshape(npts,2)
            wno = dat[:,0]
            spc = dat[:,1]
    return wno, spc

# note that read_spec fails if tab_h2o.asc is the only file in the directory.
# this might happen if you tabulate spectral coefficients before optical depth or cooling rates.

def run_RFM(par,dataset):
    # The inputs must be provided at the desired RFM resolution.
    #############################################################
    from .make_input_files import generate_atm_file, make_driver, make_kabs_driver
    from .run import run
    #############################################################
    rfmcase = par.rfmcase
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
    generate_atm_file(
        f'{rfmcase}.atm',
        heights / 1e3,  # Convert to km
        temps,
        p,
        **xgases  # Pass all detected gases' molar mixing ratios [ppmv]
    )
    #############################################################
    # Create LEV file (space-separated vertical coordinates in km)
    np.savetxt(RFM_DIR+'/lev/%s.lev'%(rfmcase), heights/1e3, delimiter=' ')
    #############################################################
    # Generate RFM driver
    # https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm    
    PHY = (      ' CPKMOL=29012.0' + '\n' +  # molar heat capacity of air (J/K/kmol)(used for COO only)
                 ' GRAVTY=9.81'    + '\n' +  # m/s^2 (ignored by HYD flag)
                 ' RADCRV=6400.'   + '\n' +  # local radius of curvature (km)
                 ' TEMSPA=2.7'     + '\n' +  # cosmic background temperature (K)
                 ' WGTAIR=28.964'            # molar mass of air (kg/kmol) (ignored by HYD flag)
          )
    print(PHY)
    # Extract active gases from the GAS list
    gases = [key[1:].upper() for key in xgases]  # e.g., ['N2', 'CH4', 'H2']
    if "-octm" in rfmcase: # option: continuum-only calculation
        gases = [f"{key[1:].upper()}(CTM)" for key in xgases]
    # Optional collision-induced absorption (CIA) files
    ciafiles = [
        f"{RFM_DIR}/hit/N2-N2_0_5000.cia",
        f"{RFM_DIR}/hit/N2-H2_0_1886.cia",
        f"{RFM_DIR}/hit/N2-CH4_0_1379.cia",
        f"{RFM_DIR}/hit/CH4-CH4_200_800.cia",
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
        'atmfile': f'{rfmcase}.atm',
        'SPC': f"{par.nu0} {par.nu1} {par.dnu}",
        'GAS': " ".join(gases),
        'HIT': f"{RFM_DIR}/hit/h2o_co2_ch4_10_1500_hitran20.par",
        'OUTDIR': f"{RFM_DIR}/outp/{rfmcase}",
        'PHY': PHY,
        'NLEV': f"{RFM_DIR}/lev/{rfmcase}.lev",
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
        'cooling': lambda: make_driver(**common_args, SFC=f"TEMREL={TEMREL:.1f}"),
        'continuum_cooling': lambda: make_driver(**common_args, SFC=f"TEMREL={TEMREL:.1f}"),
        'od_trans': lambda: make_driver(**common_args),
        'continuum_od_trans': lambda: make_driver(**common_args),
        'kabs': lambda: make_kabs_driver(**common_args, TAN="1 550 1 82 1 100"),
        'continuum_kabs': lambda: make_kabs_driver(**common_args, TAN="1 550 1 82 1 100"),
    }
    # run the appropriate driver
    if runtype in runtype_mapping:
        runtype_mapping[runtype]()
    else:
        raise ValueError(f"Unsupported runtype: {runtype}")
    #############################################################
    # Run RFM    
    print(RFM_DIR)
    run(drv_file=f"{RFM_DIR}/src/rfm.drv")
    #############################################################
    print(f'done with RFM for '+ rfmcase)
    return

def get_hr(par,dataset):
    #############################################################
    # Unpack the inputs
    rfmcase = par.rfmcase
    heights = dataset['RFM']['z']
    #############################################################
    srhr = [] # spectrally-resolved heating rate at each height (levs,nus)
    hr   = [] # spectrally-integ.   heating rate at each height (levs)
    fdir = '/home/fspauldinga/SAM24a/RFM/outp/%s'%rfmcase
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

def get_odepth(par,dataset)->dict[str,float]:
    #############################################################
    # Unpack the inputs
    rfmcase = par.rfmcase
    heights = dataset['RFM']['z']
    #############################################################
    # odepth uses an internal decision for crdnu
    crdnu = 40 # cm-1
    #############################################################
    srtau    = []  # spectrally-resolved optical depth at each height (levs,nus)
    zsrtau1  = []  # tau=1 heights as a function of nu (nus)
    D        = 1.5 # two-stream diffusivity factor
    fdir     = RFM_DIR+'/outp/%s'%rfmcase
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
    srtau = D*np.array(srtau)
    #srtau  = np.array(srtau)
      
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

def get_bint_planck(par,dataset)->Dict[str,float]:
    #############################################################
    # Unpack the inputs
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


