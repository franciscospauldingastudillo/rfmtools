"""
A.W. 2021

Set of functions to generate input files for RFM.
"""

import numpy as np
import climlab
import xarray as xr
from .utils import RFM_DIR 
import time
import os

def generate_atm_file(fname, height, temp, pres, **gases):
    """
    Generate a .atm input file for RFM.
    
    Inputs:
    * gases: molar mixing ratios in ppmv, for example,
    variables = [height, pres, temp, h2o, np.ones(nz)*co2] 
    """
    nz = int(len(height))
    
    # Standard variables
    variables = [height, pres, temp]
    labels=["*HGT [km]","*PRE [mb]","*TEM [K]"]
 
    #labels=["*HGT [km]","*PRE [mb]","*TEM [K]","*H2O [ppmv]","*CO2 [ppmv]","*CH4 [ppmv]"]
    
    # Add gases dynamically
    for gas_name, gas_profile in gases.items():
        # Ensure consistent naming: 'xCO2' → '*CO2 [ppmv]'
        gas_label = f"*{gas_name[1:].upper()} [ppmv]" if gas_name.startswith('x') else f"*{gas_name.upper()} [ppmv]"
        
        # Error checking for correct array length
        if len(gas_profile) != nz:
            raise ValueError(f"Length mismatch for {gas_name}: expected {nz}, got {len(gas_profile)}")
        
        print(f'appending {gas_label} to atm file')
        labels.append(gas_label)
        variables.append(gas_profile)
    
    with open(RFM_DIR+"/atm/"+fname,"w+") as file:
        file.write("! Produced using rfmtools - Andrew Williams 2021. \n")
        file.write("! andrew.williams@physics.ox.ac.uk \n ")
        
        file.write(f" {nz} ! No.Levels in profiles \n")
        
        for idx, label in enumerate(labels):
            file.write(f"{label} \n")
            x=",       ".join([str(i) for i in variables[idx]])
            file.write(f" {x} \n")
        file.write("*END")
    file.close()
        
        
def make_driver(
    runtype='radiance',
    extra_flags=None,
    fname=None, atmfile=None, 
    SPC="0.1 3500 0.1",
    GAS="CO2 H2O CH4",
    HIT=None,
    HDR=None,
    OUTDIR=None,
    PHY=None,
    NLEV=None,
    SFC=None,
    CIA=None):
    """
    Depending on `genre`, we'll expect a certain number of flags to be present...
    Maybe make flags an input LIST???
    
    * runtype: 'optical_depth' or 'radiance'
        sets the default flags
    
    * extra_flags: (`str`)
        For setting additional flags, e.g. 'CTM'
        
    * HYD: allows for non-hydrostatic profiles (incompatible with SFC, replace with ZEN - upward looking geometry)
           advantage: mean molecular weight is not a constant
         
    """
    import os
    if runtype=='radiance':
        flags="RAD FLX SFC"
        #flags="RAD FLX ZEN HYD"
    elif runtype=='optical_depth':
        flags="OPT"
        #flags="OPT HYD"
    elif runtype=='cooling':
        flags="RAD FLX SFC COO PRF"
        #flags="FLX ZEN COO PRF HYD"
    elif runtype=='continuum_cooling':
        flags="RAD FLX SFC COO CTM PRF"
        #flags="FLX ZEN COO CTM PRF HYD"
    elif runtype=='od_trans': # optical depth & transmittance
        flags="RAD FLX TRA OPT VRT ZEN"
        #flags="FLX TRA OPT VRT ZEN HYD"
    elif runtype=='continuum_od_trans': # optical depth & transmittance with continuum
        flags="RAD FLX TRA OPT VRT ZEN CTM"
        #flags="FLX TRA OPT VRT ZEN CTM HYD"
    else:
        raise ValueError("runtype not recognized.")
    
    print(f"flags enabled: {flags}")
    
    if fname is None:
        raise ValueError("Need to specify a filename, `fname`.") 
        
    # Check atmfile is present, and that it's an absolute path
    if atmfile is None:
        raise ValueError("Oops! You need to specify a `.atm` input file!") 
    elif not os.path.isabs(atmfile):
        atmfile=RFM_DIR+"/atm/"+atmfile
    
    # Make sure HITRAN2016 data is there
    if HIT is None:
        HIT=f"{RFM_DIR}/hit/h2o_co2_0-3500cm-1.par"
    
    # Give it a name
    if HDR is None:
        HDR="FSA2024: RFM"
    
    if OUTDIR is None:
        stamp = time.time()*1e5
        x = RFM_DIR+"/outp/"+"tmp_%.0f/" % stamp
        OUTDIR="  OUTDIR = "+x # create a unique tmp dir
        os.mkdir(x)
    else: 
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        #else: 
        #    # clean the directory as needed
        #    cmd1 = 'rm %s/*.asc'%(OUTDIR)
        #    os.system(cmd1)
        OUTDIR=" OUTDIR = "+OUTDIR+"/" # append desired directory (FSA22)
        print(OUTDIR)


    # Check PHY is present (FSA 2022)
    if PHY is None:
        raise ValueError("Oops! You need to specify a PHY input file!")     
    
    if NLEV is None:
        fp = open(atmfile)
        for i, line in enumerate(fp):
            if line[0:9]=='*PRE [mb]':
                LEV_idx = i-1
            else:
                continue
        fp.close()

        fp = open(atmfile)
        LEV = fp.readlines()[LEV_idx].split()[-1]
    else: 
        LEV = NLEV
        
    if SFC is None:
        SFC = "TEMREL=0.0"
        
    if CIA is None:
        CIA = ""
        
    # Make flags!
    labels={}
    labels["*HDR"] = HDR
    labels["*FLG"] = flags
    labels["*SPC"] = SPC
    labels["*GAS"] = GAS
    labels["*ATM"] = atmfile
    labels["*LEV"] = LEV
    labels["*HIT"] = HIT
    labels["*PHY"] = PHY
    labels["*OUT"] = OUTDIR
    labels["*SFC"] = SFC
    labels["*CIA"] = CIA
    
    # Main loop, to make .drv file
    with open(RFM_DIR+"/src/"+fname,"w+") as file:
        for idx, label in enumerate(labels.keys()):
            #print(label)
            file.write(f"{label} \n")
            file.write(f"  {labels[label]} \n")
            
        file.write("*END")
    file.close()
    
    
def make_kabs_driver(
    runtype='kabs',
    extra_flags=None,
    fname=None, atmfile=None,
    SPC="0.1 3500 0.1",
    GAS="CO2 H2O",
    HIT=None,
    HDR=None,
    OUTDIR=None,
    PHY=None,
    TAN=None):
    """
    Depending on `genre`, we'll expect a certain number of flags to be present...
    Maybe make flags an input LIST???
    
    * runtype: 'optical_depth' or 'radiance'
        sets the default flags
    
    * extra_flags: (`str`)
        For setting additional flags, e.g. 'CTM'
         
    """
    import os
    if runtype=='kabs':
        flags="TAB"
    elif runtype=='continuum_kabs':
        flags="TAB CTM"
    else:
        raise ValueError("runtype not recognized.")
        
    if fname is None:
        raise ValueError("Need to specify a filename, `fname`.") 
    
    # Check atmfile is present, and that it's an absolute path
    if atmfile is None:
        raise ValueError("Oops! You need to specify a `.atm` input file!") 
    elif not os.path.isabs(atmfile):
        atmfile=RFM_DIR+"/atm/"+atmfile
    
    # Make sure HITRAN2016 data is there
    if HIT is None:
        HIT=f"{RFM_DIR}/hit/h2o_co2_0-3500cm-1.par"
    
    # Give it a name
    if HDR is None:
        HDR="FSA2024: RFM"
    
    if OUTDIR is None:
        stamp = time.time()*1e5
        x = RFM_DIR+"/outp/"+"tmp_%.0f/" % stamp
        OUTDIR="  OUTDIR = "+x # create a unique tmp dir
        os.mkdir(x)
    else: 
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        #else: 
        #    # clean the directory as needed
        #    cmd1 = 'rm %s/*.asc'%(OUTDIR)
        #    os.system(cmd1)
        OUTDIR=" OUTDIR = "+OUTDIR+"/" # append desired directory (FSA22)
        print(OUTDIR)


    # Check PHY is present (FSA 2022)
    if PHY is None:
        raise ValueError("Oops! You need to specify a PHY input file!")     
    
    # Check TAN is present (FSA 2022)
    if TAN is None:
        raise ValueError("Oops! You need to specify TAN (in this case is interpreted as DIM)")    
        
        
    # Make flags!
    labels={}
    labels["*HDR"] = HDR
    labels["*FLG"] = flags
    labels["*SPC"] = SPC
    labels["*GAS"] = GAS
    labels["*ATM"] = atmfile
    labels["*TAN"] = TAN
    labels["*HIT"] = HIT
    labels["*PHY"] = PHY
    labels["*OUT"] = OUTDIR
    
    # Main loop, to make .drv file
    with open(RFM_DIR+"/src/"+fname,"w+") as file:
        for idx, label in enumerate(labels.keys()):
            #print(label)
            file.write(f"{label} \n")
            file.write(f"  {labels[label]} \n")
            
        file.write("*END")
    file.close()    
    
    