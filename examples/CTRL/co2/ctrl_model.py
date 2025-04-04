import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('/home/fspauldinga/SAM24a/RFM/rfmtools')
import rfmtools
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
    
##################################################################################################################
######################################### RUN RFM FOR CARBON DIOXIDE #############################################
##################################################################################################################

# Step 0: Define the case and the resolution
class params:
    def __init__(self):
        # Specify the case
        self.RHmid   = 0.75
        self.zmid    = 7.5e3
        self.uniform = True
        self.case    = 'ctrl-co2-350ppmv'
        # Specify the spectral resolution
        self.nu0      = 10   # cm-1
        self.nu1      = 1500 # cm-1
        self.dnu      = 0.1  # cm-1
        self.crdnu    = 10   # cm-1
        self.band     = 'co2' 
        self.runtype  = 'continuum_cooling' 
        self.cp       = 29012/28.964 # J/kg/K
        self.nsday    = 86400 # s
        self.TEMREL   = 0 # ground-air temp diff. (>0 means warmer surface)
        # Identify band coordinates i0,i1
        if self.band=='wv-rot':
            self.nus = np.arange(self.nu0,1000+self.dnu,self.dnu) 
            self.i0  = np.squeeze(np.where(self.nus==self.nu0))
            self.i1  = np.squeeze(np.where(np.abs(self.nus-1000)==np.min(np.abs(self.nus-1000))))
        elif self.band=='wv-vib-rot':
            self.nus = np.arange(1000,self.nu1+self.dnu,self.dnu)
            self.i0  = np.squeeze(np.where(np.abs(self.nus-1000)==np.min(np.abs(self.nus-1000))))
            self.i1  = np.squeeze(np.where(np.abs(self.nus-self.nu1)==np.min(np.abs(self.nus-self.nu1))))
        elif self.band=='wv-broadband':
            self.nus = np.arange(self.nu0,self.nu1+self.dnu,self.dnu)
            self.i0  = np.squeeze(np.where(self.nus==self.nu0))
            self.i1  = np.squeeze(np.where(np.abs(self.nus-self.nu1)==np.min(np.abs(self.nus-self.nu1))))
        elif self.band=='co2':
            self.nus = np.arange(self.nu0,self.nu1+self.dnu,self.dnu)
            self.i0  = np.squeeze(np.where(self.nus==self.nu0))
            self.i1  = np.squeeze(np.where(np.abs(self.nus-self.nu1)==np.min(np.abs(self.nus-self.nu1))))
        elif self.band=='ch4':
            self.nus = np.arange(self.nu0,self.nu1+self.dnu,self.dnu)
            self.i0  = np.squeeze(np.where(self.nus==self.nu0))
            self.i1  = np.squeeze(np.where(np.abs(self.nus-self.nu1)==np.min(np.abs(self.nus-self.nu1))))
        self.nnus = len(self.nus)
par = params()

RFM      = np.arange(0,30e3,1e2)
RFMi     = (RFM[1::]+RFM[:-1])/2
dataset  = ({'RFM':{'z':RFM,'p':{},'T':{},'rho':{},'Gamma':{},'RH':{},'hr':{},'srhr':{},
                   'srtau':{},'zsrtau1':{},'crnus':{},'crsrtau':{},'crzsrtau1':{},
                   'piBsr':{},'piBbar':{},'Trsr':{},'Tr':{}},
            'RFMi':{'z':RFMi,'p':{},'T':{},'rho':{},'Gamma':{},'RH':{}},
            'par':par}) 
         # z ~ m, p ~ Pa, T ~ K, Gamma ~ K/m, RH~unitless, hr~W/m3, srhr~cm*W/m3
         # srtau~unitless, zsrtau1~m, crnus~cm-1, crsrtau~unitless, crzsrtau1~m
         # piBsr~W*cm/m^2, piBbar~W/m^2

# Step 1: Generate custom atmospheric profiles (Pa,K,m) at RFM and RFMi resolution
from ctrl_data import get_custom_atm
dat1,dat2 = get_custom_atm(par.RHmid,par.zmid,par.uniform,RFM,RFMi)
#
dataset['RFM']['p']      = dat1['p']
dataset['RFM']['T']      = dat1['T']
dataset['RFM']['RH']     = dat1['RH']
dataset['RFM']['rho']    = dat1['rho']
dataset['RFM']['Gamma']  = dat1['Gamma']
#
dataset['RFMi']['p']     = dat2['p']
dataset['RFMi']['T']     = dat2['T']
dataset['RFMi']['RH']    = dat2['RH']
dataset['RFMi']['rho']   = dat2['rho']
dataset['RFMi']['Gamma'] = dat2['Gamma']
#
print('done with step 1.\n')


# Step 2: Run RFM for cooling rates in K/day
from ctrl_data import run_RFM_CO2
par.runtype    = "cooling" 
dataset['par'] = par # updating par.runtype
print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
dat            = run_RFM_CO2(dataset)
print('done with step 2.\n')

# Step 3: Run RFM for optical depth and transmissivity
par.runtype    = "od_trans" 
dataset['par'] = par
print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
dat            = run_RFM_CO2(dataset)
print('done with step 3.\n')

# Step 4: Run RFM for reference absorption coefficient distribution
par.runtype    = "kabs" 
dataset['par'] = par 
print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
dat            = run_RFM_CO2(dataset)
print('done with step 4.\n')

# Sanity check: cooling rate in K/day
from ctrl_data import get_hr
par.runtype    = "cooling" 
dataset['par'] = par 
dat            = get_hr(dataset) # RFM default is K/day
fig,ax         = plt.subplots(figsize=(dim,dim))
ax.plot(dat['hr'],dataset['RFM']['z']/1e3,label='co2 w/o ctm')
ax.legend()
ax.set_xlabel('cooling rate (K/day)')
ax.set_ylabel('height (km)')
ax.set_ylim([0,20])
ax.set_xlim([-1,2])
path = rfmtools.utils.RFM_DIR+'/rfmtools/examples/CTRL/co2'
filename = '%s/%s.pdf' % (path,par.case)
plt.savefig(filename,dpi=200)  