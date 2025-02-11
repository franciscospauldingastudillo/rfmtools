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
    
def get_custom_atm(RHmid=0.5,zmid=7.5e3,uniform=False,vres1=np.arange(0,30e3,1e2),vres2=np.arange(0,30e3,1e2)):
    # scalar inputs: mid-tropospheric RH (RHmid) and height (zmid) and boolean option for uniform RH
    # Vertical resolution can also be specified to facilitate RFM vs analytical model. 
    #############################################################
    # Define a parameter class with thermodynamic constants
    class params:
        def __init__(self):
            self.Ra   = 287
            self.Rv   = 462
            self.cpd  = 1004
            self.cpv  = 1880
            self.L    = 2.5e6
            self.g    = 9.81
            self.Ts   = 300
            self.Ttrp = 200
            self.Gamma= 7e-3
            self.ps   = 1e5
            self.xi   = 1
            self.alpha = 0.25
            self.zs    = 0
            self.ztop  = 30e3
            self.z     = np.arange(0,3e4,1)
    #############################################################
    def get_custom_RH(par,RHs,RHmid,RHtrp,zs=0,zmid=7.5e3,ztrp=15e3):
        # We fit an exponential function between the surface, mid-troposphere, and tropopause
        # for prescribed values of RH and mid-tropospheric height. 
        def calculate_exponential_coefficients(x1, y1, x2, y2):
            b = np.log(y2 / y1) / (x2 - x1)
            a = y1 / np.exp(b * x1)
            return a, b
        def exponential_function(x, a, b):
            return a * np.exp(b * x)

        # Lower Troposphere
        x1 = zs
        y1 = RHs
        x2 = zmid
        y2 = RHmid
        # Upper Troposphere
        x2 = zmid
        y2 = RHmid
        x3 = ztrp
        y3 = RHtrp

        # Calculate coefficients 'a' and 'b'
        a, b = calculate_exponential_coefficients(x1, y1, x2, y2)
        c, d = calculate_exponential_coefficients(x2, y2, x3, y3)

        RH = np.zeros([len(par.z)],dtype='float')
        for k,z in enumerate(par.z):
            if z<=zmid:
                RH[k] = exponential_function(z,a,b)
            elif z<=ztrp:
                RH[k] = exponential_function(z,c,d)
            else:
                RH[k] = RHtrp
                
        # A least-squares polynomial fit is used to smooth RH (deg=15)
        lsfit   = np.poly1d(np.polyfit(par.z,RH,15))
        RHls    = lsfit(par.z)
        RHls    = np.where(par.z>18e3,0.75,RHls)
                
        return RHls
    #############################################################
    par = params()
    #############################################################
    # Custom Gamma
    Gamma = np.ones([len(par.z)],dtype='float')*par.Gamma
    # Custom T
    T  = par.Ts - par.Gamma*par.z
    T  = np.where(T<par.Ttrp,par.Ttrp,T)
    # Identify the height of the tropopause
    ktrp = int(np.amin(np.where(T==par.Ttrp)))
    ztrp = par.z[ktrp]
    # Custom RH
    if uniform is True:
        RH = get_custom_RH(par,RHmid,RHmid,RHmid,par.zs,zmid,ztrp)
    else:
        RH = get_custom_RH(par,0.75,RHmid,0.75,par.zs,zmid,ztrp)
    # Custom p (Pa)
    p  = np.ones([len(par.z)],dtype='float')*par.ps
    arg   = 0.
    for k in range(len(par.z)-1):
        dz = par.z[k+1]-par.z[k]
        arg+=-par.g/par.Ra*dz/T[k]
        p[k+1] = par.ps*np.exp(arg)
    # Custom rho
    rho    = np.zeros([len(par.z)],dtype='float')
    rho[:] = p[:]/(par.Ra*T[:])
    #############################################################
    # Export fields in their desired vertical resolution (vres1)
    T1    = np.interp(vres1,par.z,T)
    RH1   = np.interp(vres1,par.z,RH)
    p1    = np.interp(vres1,par.z,p)
    Gamma1= np.interp(vres1,par.z,Gamma)
    rho1  = np.interp(vres1,par.z,rho)
    dat1  = {'T':T1,'RH':RH1,'p':p1,'Gamma':Gamma1,'rho':rho1}
    #############################################################
    # Export fields in their desired vertical resolution (vres2)
    T2    = np.interp(vres2,par.z,T)
    RH2   = np.interp(vres2,par.z,RH)
    p2    = np.interp(vres2,par.z,p)
    Gamma2= np.interp(vres2,par.z,Gamma)
    rho2  = np.interp(vres2,par.z,rho)
    dat2  = {'T':T2,'RH':RH2,'p':p2,'Gamma':Gamma2,'rho':rho2}
    #############################################################
    # T(K), RH(unitless), p(Pa), Gamma(K/m), rho(kg/m3)
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


def run_RFM_H2O(dataset):
    # The inputs must be provided at the desired RFM resolution.
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    case    = par.case
    runtype = par.runtype
    #############################################################
    # TEMREL (sea-air temp difference (>0 means warmer surface))
    TEMREL = par.TEMREL
    #############################################################
    # Height (increasing left to right, m)
    heights = dataset['RFM']['z']
    #############################################################
    # Wavenumbers (cm-1)
    nus = par.nus
    #############################################################
    # Temperature (K)
    temps = dataset['RFM']['T']
    #############################################################
    # Total pressure (*hPa*)
    p = dataset['RFM']['p']/1e2
    #############################################################
    # Tropopause height and vertical index
    ktrp      = np.amin(np.where(temps==200))
    ztrp      = heights[ktrp]
    #############################################################
    # H2O volumetric mixing ratio (# mol h2o/total # mol, ppmv)
    RH     = dataset['RFM']['RH']
    e      = RH * climlab.utils.thermo.clausius_clapeyron(temps) # (*hPa*)
    h2ovmr = e/p * 1.e6 # ppmv
    h2ovmr = np.where(heights>=ztrp,h2ovmr[ktrp],h2ovmr) # stratospheric wv pegged to tropopause
    #############################################################
    # CO2 volumetric mixing ratio (# mol h2o/total # mol, ppmv)
    co2vmr = np.ones([len(h2ovmr)],dtype='float')*350.*0.0 # ppmv
    #############################################################
    # CH4 volumetric mixing ratio (# mol co2/total # mol, ppmv)
    ch4vmr = np.ones([len(h2ovmr)],dtype='float')*0.0 # ppmv
    #############################################################
    # Generate RFM input file: height (km), temps (K), p (hPa), 
    # h2ovmr (array ppmv of H2O), co2vmr (array ppmv of CO2)
    rfmtools.make_input_files.generate_atm_file('%s.atm'%(case), heights/1e3, temps, p, h2o=h2ovmr, co2=co2vmr, ch4=ch4vmr)
    #############################################################
    # Create LEV file (space-separated vertical coordinates in km)
    np.savetxt(rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case), heights/1e3, delimiter=' ')
    #############################################################
    # Generate RFM driver
    # https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
    constants = (' CPKMOL=29012.0' + '\n' +  # molar heat capacity of air (J/K/kmol)
                 ' GRAVTY=9.81'    + '\n' +  # m/s^2
                 ' RADCRV=6400.'   + '\n' +  # local radius of curvature (km)
                 ' TEMSPA=2.7'     + '\n' +  # cosmic background temperature (K)
                 ' WGTAIR=28.964')           # molar mass of air (kg/kmol)
    print(constants)
    if runtype=='cooling':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,   # WV continuum: "cooling"(off) or "continuum_cooling"(on)
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case),
            SFC="TEMREL=%.1f"%(TEMREL))
    elif runtype=='continuum_cooling':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,   # WV continuum: "cooling"(off) or "continuum_cooling"(on)
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case),
            SFC="TEMREL=%.1f"%(TEMREL))
    elif runtype=='od_trans':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case))
    elif runtype=='continuum_od_trans':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case))
    elif runtype=='kabs':
        rfmtools.make_input_files.make_kabs_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case),
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            TAN="1 500 1 260 1 100")
            #TAN="PLV 180 320 10") # use p,T values from atmfile for the look-up table
            #TAN="/home/fspauldinga/RFM/dat/pfile.dat\n/home/fspauldinga/RFM/dat/Tfile.dat") # T,p values for the look-up table
    elif runtype=='continuum_kabs':
        rfmtools.make_input_files.make_kabs_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case),
            SPC="10 1500 0.1", # spectral range (cm-1)
            GAS="H2O",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            TAN="1 500 1 260 1 100")
    #############################################################
    # Run RFM
    print(rfmtools.utils.RFM_DIR)
    rfmrun = rfmtools.run.run(drv_file=rfmtools.utils.RFM_DIR+"/src/rfm.drv")
    #############################################################
    print('done with RFM for H2O.')
    return
            
def run_RFM_CO2(dataset):
    # The inputs must be provided at the desired RFM resolution.
    #############################################################
    # Unpack the inputs
    par     = dataset['par']
    case    = par.case
    runtype = par.runtype
    #############################################################
    # TEMREL (sea-air temp difference (>0 means warmer surface))
    TEMREL = par.TEMREL
    #############################################################
    # Height (increasing left to right, m)
    heights = dataset['RFM']['z']
    #############################################################
    # Wavenumbers (cm-1)
    nus = par.nus
    #############################################################
    # Temperature (K)
    temps = dataset['RFM']['T']
    #############################################################
    # Total pressure (*hPa*)
    p = dataset['RFM']['p']/1e2
    #############################################################
    # Tropopause height and vertical index
    ktrp      = np.amin(np.where(temps==200))
    ztrp      = heights[ktrp]
    #############################################################
    # H2O volumetric mixing ratio (# mol h2o/total # mol, ppmv)
    RH     = dataset['RFM']['RH']
    e      = RH * climlab.utils.thermo.clausius_clapeyron(temps) # (*hPa*)
    h2ovmr = e/p * 1.e6 * 0.0 # ppmv
    #h2ovmr = np.where(heights>=ztrp,h2ovmr[ktrp],h2ovmr) # stratospheric wv pegged to tropopause
    #############################################################
    # CO2 volumetric mixing ratio (# mol h2o/total # mol, ppmv)
    co2vmr = np.ones([len(h2ovmr)],dtype='float')*350. # ppmv
    #############################################################
    # CH4 volumetric mixing ratio (# mol co2/total # mol, ppmv)
    ch4vmr = np.ones([len(h2ovmr)],dtype='float')*0.0 # ppmv
    #############################################################
    # Generate RFM input file: height (km), temps (K), p (hPa), 
    # h2ovmr (array ppmv of H2O), co2vmr (array ppmv of CO2)
    rfmtools.make_input_files.generate_atm_file('%s.atm'%(case), heights/1e3, temps, p, h2o=h2ovmr, co2=co2vmr, ch4=ch4vmr)
    #############################################################
    # Create LEV file (space-separated vertical coordinates in km)
    np.savetxt(rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case), heights/1e3, delimiter=' ')
    #############################################################
    # Generate RFM driver
    # https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
    constants = (' CPKMOL=29012.0' + '\n' +  # molar heat capacity of air (J/K/kmol)
                 ' GRAVTY=9.81'    + '\n' +  # m/s^2
                 ' RADCRV=6400.'   + '\n' +  # local radius of curvature (km)
                 ' TEMSPA=2.7'     + '\n' +  # cosmic background temperature (K)
                 ' WGTAIR=28.964')           # molar mass of air (kg/kmol)
    print(constants)
    if runtype=='cooling':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,   # WV continuum: "cooling"(off) or "continuum_cooling"(on)
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case),
            SFC="TEMREL=%.1f"%(TEMREL))
    elif runtype=='continuum_cooling':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,   # WV continuum: "cooling"(off) or "continuum_cooling"(on)
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case),
            SFC="TEMREL=%.1f"%(TEMREL))
    elif runtype=='od_trans':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case))
    elif runtype=='continuum_od_trans':
        rfmtools.make_input_files.make_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case), 
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="H2O CO2 CH4",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            NLEV=rfmtools.utils.RFM_DIR+'/lev/%s.lev'%(case))   
    elif runtype=='kabs':
        rfmtools.make_input_files.make_kabs_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case),
            SPC=f"{par.nu0} {par.nu1} {par.dnu}", # spectral range (cm-1)
            GAS="CO2",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            TAN="1 500 1 260 1 100")
            #TAN="PLV 180 320 10") # use p,T values from atmfile for the look-up table
            #TAN="/home/fspauldinga/RFM/dat/pfile.dat\n/home/fspauldinga/RFM/dat/Tfile.dat") # T,p values for the look-up table
    elif runtype=='continuum_kabs':
        rfmtools.make_input_files.make_kabs_driver(
            runtype=runtype,  
            fname='rfm.drv',
            atmfile='%s.atm'%(case),
            SPC="10 1500 0.1", # spectral range (cm-1)
            GAS="CO2",
            HIT=rfmtools.utils.RFM_DIR+"/hit/h2o_co2_ch4_10_1500_hitran20.par", # hitran coefficients
            OUTDIR=rfmtools.utils.RFM_DIR+'/outp/%s'%(case),
            PHY=constants,
            TAN="1 500 1 260 1 100")
    #############################################################
    # Run RFM
    print(rfmtools.utils.RFM_DIR)
    rfmrun = rfmtools.run.run(drv_file=rfmtools.utils.RFM_DIR+"/src/rfm.drv")
    #############################################################
    print('done with RFM for CO2.')
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
        
    return {'Trsr':Trsr,'Tr':Tr,'dpTrsr':dpTrsr,'dpTr':dpTr,'dzTrsr':dzTrsr,'dzTr':dzTr} 