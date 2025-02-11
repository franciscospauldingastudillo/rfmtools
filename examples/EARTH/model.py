# load dependencies
from data import *
    
##################################################################################################################
######################################### RUN RFM FOR EARTH  #####################################################
##################################################################################################################

# Step 0: Define the case and the resolution
class parameters:
    def __init__(self,**params):
        # Spectral resolution used in RFM experiments
        self.nu0 = params.get('nu0', 10)  # cm-1
        self.nu1 = params.get('nu1', 1500)  # cm-1
        self.dnu = params.get('dnu', 0.1)  # cm-1
        self.band = params.get('band', 'earth')
        self.runtype = params.get('runtype', 'cooling')
        self.cpdef = params.get('cp', 29012 / 28.964)  # J/kg/K (default RFM value of specific heat)
        self.nsday = params.get('nsday', 86400)  # seconds per Earth-day
        self.TEMREL = params.get('TEMREL', 0)  # ground-air temp diff.

        # spectral range calculations
        self.nus = np.arange(self.nu0, self.nu1 + self.dnu, self.dnu)
        if self.band == 'earth':
            self.i0 = np.squeeze(np.where(np.abs(self.nus - self.nu0) == np.min(np.abs(self.nus - self.nu0))))
            self.i1 = np.squeeze(np.where(np.abs(self.nus - self.nu1) == np.min(np.abs(self.nus - self.nu1))))
        else:
            raise ValueError(f"Invalid band type: {self.band}. Expected 'earth'.")
        self.nnus = len(self.nus)

        # Thermodynamic parameters
        self.ps = params.get('ps', 1e5)
        self.Ts = params.get('Ts', 300)
        self.Ttrp = params.get('Ttrp', 200)
        self.Gamma = params.get('Gamma', 6.5e-3)
        self.z = params.get('z', np.arange(0, 3e4, 1))

        # Saturation Vapor Pressure (Water Vapor)
        self.Ttrip = params.get('Ttrip', 273.16)  # K 
        self.ptrip = params.get('ptrip', 611.65)  # Pa
        self.E0v   = params.get('E0v', 2.3740e6)  # J/kg 
        self.ggr   = params.get('ggr', 9.81)      # m/s^2, gravity
        self.rgasa = params.get('rgasa', 287.04)  # J/kg/K, specific gas constant of dry air
        self.rgasv = params.get('rgasv', 461.5)   # J/kg/K, specific gas constant of water vapor
        self.cva   = params.get('cva', 719)       # J/kg/K, isovolumetric specific heat of dry air
        self.cvv   = params.get('cvv', 1418)      # J/kg/K, isovolumetric specific heat of water vapor
        self.cvl   = params.get('cvl', 4119)      # J/kg/K, isovolumetric specific heat of liquid water
        self.cpa   = params.get('cpa', self.cva + self.rgasa)   # isobaric specific heat of dry air
        self.cpv   = params.get('cpv', self.cvv + self.rgasv)   # isobaric specific heat of water vapor
        self.eps   = params.get('eps', self.rgasa / self.rgasv) # ratio of specific gas constants
        self.L     = params.get('L', 2.5e6)       # enthalpy of vaporization of water
        self.E0s   = params.get('E0s', np.nan)    # no ice phase
        self.cvs   = params.get('cvs', np.nan)    # no ice phase
           
        # Earth-like composition parameters
        self.Runi  = params.get('Runi',8.314)               # universal gas constant (J/mol/K)
        self.MN2   = params.get('MN2',2*14.01*1e-3)         # molecular weight of N2 (kg/mol)
        self.MO2   = params.get('MO2',2*15.999*1e-3)        # molecular weight of O2 (kg/mol)
        self.MH2O  = params.get('MH2O',18.015*1e-3)         # molecular weight of H2O (kg/mol)
        self.MCO2  = params.get('MCO2',44.009*1e-3)         # molecular weight of CO2 (kg/mol)
        self.RN2   = params.get('RN2',self.Runi/self.MN2)   # specific gas constant of N2 (J/kg/K)
        self.RO2   = params.get('RO2',self.Runi/self.MO2)   # specific gas constant of O2 (J/kg/K)
        self.RH2O  = params.get('RH2O',self.Runi/self.MH2O) # specific gas constant of H2O (J/kg/K)
        self.RCO2  = params.get('RCO2',self.Runi/self.MCO2) # specific gas constant of CO2 (J/kg/K)
        
        # Earth-like molar mixing ratios
        self.xCO2  = params.get('xCO2',395e-6)         # molar mixing ratio of CO2 (mol/mol) ~ 395 ppmv
        self.xN2   = params.get('xN2',0.78)            # molar mixing ratio of N2 (mol/mol)  ~ 78% (dry-air approx)
        self.xO2   = params.get('xO2',0.22)            # molar mixing ratio of O2 (mol/mol)  ~ 22% (dry-air approx)
           
        # Initialize case string
        self.case = ""
        
    def generate_case(self,**params):
        planet       = params.get('planet','unknown')
        self.RHs     = params.get('RHs',0.75)
        self.RHmid   = params.get('RHmid',0.54)
        self.RHtrp   = params.get('RHtrp',0.75)
        self.Tmid    = params.get('Tmid',250)
        self.uniform = params.get('uniform',1)
        # Extract gases dynamically (gas1, gas2, ..., gasN)
        gases = [params[key] for key in sorted(params) if key.startswith('gas')]
        # Define valid CIA pairs as tuples for safety with multi-character molecules
        self.valid_ciapairs = params.get('valid_ciapairs', [('N2', 'N2'), ('N2', 'CH4'), ('N2', 'H2'), ('CH4', 'CH4')])
        # Filter CIA pairs to include only those with gases in the `gases` list
        ciapairs = [
            f"{mol1}{mol2}" for (mol1, mol2) in self.valid_ciapairs
            if mol1 in gases and mol2 in gases
        ]
        # Format the CIA part of the case name
        cia_str = "-CIA-" + "-".join(ciapairs) if ciapairs else ""
        # Construct the case name
        self.case = '-'.join([
            planet,
            *gases,
            str(int(self.RHmid * 100)),
            str(self.Tmid),
            str(self.uniform)
        ]) + cia_str
        print('generate_case: ',self.case)
        
# specify what gases and CIA-pairs to include (see generate_case)
gases     = ['H2O']
ciapairs  = []

# dynamically create an argument dictionary
def generate_args(planet, gases, ciapairs, **relhum):
    return {'planet': planet, **{f'gas{i+1}': gas for i, gas in enumerate(gases)}, 'valid_ciapairs': ciapairs, **relhum}
args = generate_args('earth', gases, ciapairs, RHs=0.75, RHmid=0.54, RHtrp=0.75, uniform=1)
        
# create a class instance and generate an RFM case from argument dictionary
par = parameters()
par.generate_case(**args)

# vertical resolutions
RFM      = np.arange(0,30e3,1e2)
RFMi     = (RFM[1::]+RFM[:-1])/2

# default dataset
# z ~ m, p ~ Pa, T ~ K, Gamma ~ K/m, RH~unitless, hr~W/m3, srhr~cm*W/m3
# srtau~unitless, zsrtau1~m, crnus~cm-1, crsrtau~unitless, crzsrtau1~m
# piBsr~W*cm/m^2, piBbar~W/m^2
dataset  = ({'RFM':{'z':RFM,'p':{},'T':{},'rho':{},'Gamma':{},'RH':{},'hr':{},'srhr':{},
                   'srtau':{},'zsrtau1':{},'crnus':{},'crsrtau':{},'crzsrtau1':{},
                   'piBsr':{},'piBbar':{},'Trsr':{},'Tr':{}},
            'par':par}) 

# Step 1: Generate custom atmospheric profiles (p,T,z,x) at RFM and RFMi resolution
dat1,dat2 = get_custom_atm(par,RFM,RFMi)
#
dataset['RFM']['p']      = dat1['p']
dataset['RFM']['T']      = dat1['T']
dataset['RFM']['RH']     = dat1['RH']
dataset['RFM']['rho']    = dat1['rho']
dataset['RFM']['Gamma']  = dat1['Gamma']

# dynamically add molar mixing ratios to the dataset (signals to rfmtools how to build experiments)
for gas in gases:
    xgas_key = f'x{gas}'  # e.g., xN2, xCH4, xH2
    dataset['RFM'][xgas_key] = dat1[xgas_key]
    
# save the thermodynamic profiles for later use
with open(f'{par.case}.pickle', 'wb') as file:
    pickle.dump(dataset, file)  
print('done with step 1.\n')

if 1:
    # Step 2: Run RFM for cooling rates in K/day
    par.runtype    = "continuum_cooling" 
    dataset['par'] = par # updating par.runtype
    print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
    dat            = run_RFM(dataset)
    print('done with step 2.\n')
if 0:
    # Step 3: Run RFM for optical depth and transmissivity
    par.runtype    = "od_trans" 
    dataset['par'] = par
    print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
    dat            = run_RFM(dataset)
    print('done with step 3.\n')
if 0:
    # Step 4: Run RFM for reference absorption coefficient distribution
    par.runtype    = "kabs" 
    dataset['par'] = par 
    print('initializing RFM in *%s* configuration'%(dataset['par'].runtype))
    dat            = run_RFM(dataset)
    print('done with step 4.\n')

# Sanity check: cooling rate in K/day
par.runtype    = "cooling" 
dataset['par'] = par # update par.runtype
dat            = get_hr(dataset) # RFM default is K/day
fig,ax         = plt.subplots(figsize=(dim,dim))
ax.plot(dat['hr'],dataset['RFM']['z']/1e3,label=f'{par.case}')
ax.legend()
ax.set_xlabel('cooling rate (K/day)')
ax.set_ylabel('height (km)')
ax.set_ylim([0,50])
ax.set_xlim([0,5])
path = rfmtools.utils.RFM_DIR+'/rfmtools/examples/EARTH'
filename = '%s/%s.pdf' % (path,par.case)
plt.savefig(filename,dpi=200)  

# a potential issue is that nitrogen/methane/hydrogen are relevant to constructing thermodynamic profiles, 
# but should not be included as radiative species unless they are in the gases list. This is a rfmtools issue?