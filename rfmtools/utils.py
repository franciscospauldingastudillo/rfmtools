import numpy as np

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
    rfmtools.make_input_files.generate_atm_file(
        f'{rfmcase}.atm',
        heights / 1e3,  # Convert to km
        temps,
        p,
        **xgases  # Pass all detected gases' molar mixing ratios [ppmv]
    )
    #############################################################
    # Create LEV file (space-separated vertical coordinates in km)
    np.savetxt(rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(rfmcase), heights/1e3, delimiter=' ')
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
        'atmfile': f'{rfmcase}.atm',
        'SPC': f"{par.nu0} {par.nu1} {par.dnu}",
        'GAS': " ".join(gases),
        'HIT': f"{rfmtools.utils.RFM_DIR}/hit/h2o_co2_ch4_10_1500_hitran20.par",
        'OUTDIR': f"{rfmtools.utils.RFM_DIR}/outp/{rfmcase}",
        'PHY': PHY,
        'NLEV': f"{rfmtools.utils.RFM_DIR}/lev/{rfmcase}.lev",
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
    print(f'done with RFM for '+ rfmcase)
    return


