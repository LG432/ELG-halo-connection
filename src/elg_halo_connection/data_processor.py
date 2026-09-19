import numpy as np
import tqdm
import os
import pickle
from scipy.spatial import cKDTree
import multiprocessing as mp

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

import illustris_python as il

# Multiprocessing worker
_worker_processor = None

def init_worker(processor):
    global _worker_processor
    _worker_processor = processor

def compute_aperture_worker(args):
    idx, snapNum, simName, R = args
    return _worker_processor.compute_aperture(idx=idx, snapNum=snapNum, simName=simName, R=R)


class DataProcessor:
    def __init__(self, simInfo, pickle_dir, minStellPart, save_data = False, overwrite = False):
        """
        simInfo is a list where each element is a tuple containing (simName, snapNum, simPath)
        cosmo is the set cosmology (Planck 2015)
        pickle_dir is the directory for storing the pickled files
        minStellPart is the minimum number of star particles in each subhalo for it to be sufficiently resolved
        
        """
        self.simInfo    = simInfo
        self.pickle_dir = pickle_dir
        self.cosmo      = cosmo
        self.minStellPart = minStellPart

        self.stellPartsDict = {}
        self.subhaloDict    = {}
        self.simDict        = {}
        self.fofDict        = {}

        # KDTree
        self.starTree     = None
        self.tree_simName = None
        self.tree_snapNum = None

        #
        self.save_data = save_data
        self.overwrite = overwrite

    def load_data(self):
        """Loads the simulation headers, star particles, subhalos and FoF halo attributes and stores each as a dictionary"""

        simInfo   = self.simInfo
        save_data = self.save_data
        overwrite = self.overwrite
        
        for simName, snapNum, simPath in simInfo:
            
            if simName not in self.stellPartsDict:
                self.stellPartsDict[simName] = {}
                self.subhaloDict[simName]    = {}
                self.simDict[simName]        = {}
                self.fofDict[simName]        = {}
    
            fnames = { 
            'stars': f'stellParts_{simName}_snap{snapNum}',
            'subs': f'subhalos_{simName}_snap{snapNum}',
            'headers': f'headers_{simName}_snap{snapNum}',
            'fofs': f'fofs_{simName}_snap{snapNum}'
            }
    
            paths = {key: os.path.join(self.pickle_dir, fname) for key, fname in fnames.items()}
    
            # Check whether all paths exist
            all_exist = all(os.path.exists(path) for path in paths.values())
    
            if all_exist and not overwrite:
                print(f"Opening saved data for {simName}, snapshot {snapNum}...")
    
                with open(paths['stars'], 'rb') as handle:
                    stars = pickle.load(handle)
    
                with open(paths['subs'], 'rb') as handle:
                    subs = pickle.load(handle)
    
                with open(paths['headers'], 'rb') as handle:
                    headers = pickle.load(handle)
    
                with open(paths['fofs'], 'rb') as handle:
                    fofs = pickle.load(handle)
    
            else:
                print(f"Loading directly from the simulation for {simName}, snapshot {snapNum}...")
            
                headers = il.groupcat.loadHeader(simPath, snapNum)
                stars   = il.snapshot.loadSubset(simPath, snapNum, 'stars', ['Coordinates', 'Masses', 'GFM_StellarFormationTime', 'GFM_Metallicity', 'GFM_InitialMass'])
                subs    = il.groupcat.loadSubhalos(simPath, snapNum, fields = ['SubhaloPos', 
                                                                               'SubhaloMassType', 
                                                                               'SubhaloHalfmassRadType', 
                                                                               'SubhaloGasMetallicity', 
                                                                               'SubhaloSFRinRad', 
                                                                               'SubhaloMassInHalfRadType', 
                                                                               'SubhaloMassInRadType',
                                                                               'SubhaloGrNr',
                                                                               'SubhaloLenType'])
                
                fofs = il.groupcat.loadHalos(simPath, snapNum, fields = ['GroupFirstSub', 'GroupNsubs', 'Group_M_Crit200', 'Group_R_Crit200'])
    
    
                if save_data:
                    
                    data = {
                            paths['stars']:stars,
                            paths['subs']: subs,
                            paths['headers']: headers,
                            paths['fofs']: fofs
                    }
                    
                    for full_path, data_dict in data.items():
                        
                        with open(full_path, 'wb') as handle:
                            pickle.dump(data_dict, handle)
    
                        print(f'Saved {full_path}')

            self.stellPartsDict[simName][snapNum] = stars
            self.subhaloDict[simName][snapNum]    = subs
            self.simDict[simName][snapNum]        = headers
            self.fofDict[simName][snapNum]        = fofs

        print("Finished loading data")

        return self.stellPartsDict, self.subhaloDict, self.simDict, self.fofDict

    def build_star_tree(self, simName, snapNum):
        """
        Builds a KDTree for all star particles in the simulation volume
        
        Coordinates are converted from ckpc h^-1 to pkpc
        
        """

        props_par = self.stellPartsDict[simName][snapNum]
        props_sim = self.simDict[simName][snapNum]

        h        = props_sim['HubbleParam']
        redshift = props_sim['Redshift']
        boxsize_co_h = props_sim['BoxSize'] * u.kpc

        # Simulation snapshot expansion factor
        scaleFac = 1 / (1 + redshift)

        # Star particle comoving coordinates ckpc h^-1
        starPos_co_h  = props_par['Coordinates'] * u.kpc

        # Star particle proper coordinates pkpc
        starPos_pr    = starPos_co_h * scaleFac / h

        # Proper boxsize kpc
        boxsize_pr = boxsize_co_h * scaleFac / h

        # Build KDTree and account for periodic boundary conditions by passing boxsize as an argument
        self.starTree = cKDTree(starPos_pr.to_value(u.kpc), boxsize_pr.to_value(u.kpc))

        self.tree_simName = simName
        self.tree_snapNum = snapNum

        print(f"Built stellar KDTree for {simName}, snapshot {snapNum}")

    def identify_subhalo_idxs(self, simName, snapNum):
        props_sub = self.subhaloDict[simName][snapNum]

        numStellParts = props_sub['SubhaloLenType'][:, 4]

        print("Number of subhalos: ", len(numStellParts))
        print("Number of selected subhalos: ", np.sum(numStellParts >= self.minStellPart))

        minStellPart  = self.minStellPart
        valid_idxs    = np.where(numStellParts >= minStellPart)[0]

        print(f"Identified subhalos with a minimum stellar particle number of {minStellPart}")

        return valid_idxs
        
    def compute_aperture(self, idx, snapNum, simName, R=30*u.kpc):
        """Compute subhalo quantities within R pkpc for a single subhalo."""
        """All positions, masses originally in comoving h^-1, we convert to proper coordinates"""
        """All particle data is for stellar particles only"""
        """Subhalo data contains information in all particle types, stellar particles are under array 4, wind phase cells are under array 0"""

        if simName not in self.subhaloDict:
            raise ValueError(f"{simName} has not been loaded")

        if snapNum not in self.subhaloDict[simName]:
            raise ValueError(f"Snapshot {snapNum} has not been loaded for {simName}")

        if self.starTree is None or self.tree_simName != simName or self.tree_snapNum != snapNum:
            raise RuntimeError(f"The stellar KDTree has not been built for {simName} at snapshot {snapNum}. Call build_star_tree() first.")
        
        # Define dictionaries for subhalos, particles, fof groups and simulation properties
        props_sub  = self.subhaloDict[simName][snapNum]    # Subhalo properties
        props_par  = self.stellPartsDict[simName][snapNum] # Particle properties
        props_fof  = self.fofDict[simName][snapNum]        # FoF group properties
        props_sim  = self.simDict[simName][snapNum]        # Simulation properties
    
        # Simulation properties
        h        = props_sim['HubbleParam'] # Little h value from simulation 
        redshift = props_sim['Redshift']    # Redshift of simulation snapshot
        scaleFac = 1 / (1 + redshift)       # Scale factor of simulation snapshot
    
        # FoF properties
        GrNr = props_sub['SubhaloGrNr'][idx]
        central_sub_idx = props_fof['GroupFirstSub'][GrNr]    
        fofM200      = props_fof['Group_M_Crit200'][GrNr] * 1e10 * u.Msun / h
        fofR200Co    = props_fof['Group_R_Crit200'][GrNr] * u.kpc / h
        fofR200Pr    = fofR200Co * scaleFac
    
        # Determine whether subhalo is a central or satellite galaxy
        isCentral = (idx == central_sub_idx)
        
        # 2. Extract Subhalo Properties
        subPosCo                  = props_sub['SubhaloPos'][idx] * u.kpc / h                      # Subhalo position (Comoving)
        subPosPr                  = subPosCo * scaleFac                                           # Subhalo position (Proper)
        subSfrIn2HfStellRad       = props_sub['SubhaloSFRinRad'][idx] * u.Msun / u.yr             # Subhalo total SFR of all gas cells within twice half stellar mass radius
        subHfStellMassRadCo       = props_sub['SubhaloHalfmassRadType'][idx, 4] * u.kpc / h       # Subhalo half stellar mass radius (Comoving)
        subHfStellMassRadPr       = subHfStellMassRadCo * scaleFac                                # Subhalo half stellar mass radius (Proper)
        subGasMetallicity         = props_sub['SubhaloGasMetallicity'][idx]                       # Subhalo mass-weighted metallicity of gas cells within twice half stellar mass radius (Absolute)
        subTotStellMass           = props_sub['SubhaloMassType'][idx, 4] * 1e10 * u.Msun / h      # Subhalo mass sum of all bound stellar particles 
        subStellMassIn2HfStellRad = props_sub['SubhaloMassInRadType'][idx, 4] * 1e10 * u.Msun / h # Subhalo mass sum of all stellar particles within twice half stellar mass radius
        
        # 3. Query Particles
        starTree  = self.starTree
        rRIdxs    = starTree.query_ball_point(subPosPr.to_value(u.kpc), R.to_value(u.kpc))
        
        # 4. Extract Particle Data
        """GFM_Metallicity is absolute (non-solar units)"""
        """GFM_StellarFormationTime is an expansion factor and can be negative for wind phase gas cells"""
        """GFM_InitialMass is the mass of the SSP at time of birth"""
        partPosCo            = props_par['Coordinates'][rRIdxs] * u.kpc / h             # Particle position within simulation (comoving)
        partPosPr            = partPosCo * scaleFac                                     # Particle position within simulation (proper)
        partMass             = props_par['Masses'][rRIdxs] * 1e10 * u.Msun / h          # Particle mass at simulation snapshot
        partBirthScalefac    = props_par['GFM_StellarFormationTime'][rRIdxs]            # Particle birth time (Scale factor)
        partBirthMetallicity = props_par['GFM_Metallicity'][rRIdxs]                     # Particle birth metallicity (Absolute)
        partBirthMass        = props_par['GFM_InitialMass'][rRIdxs] * 1e10 * u.Msun / h # Particle birth mass
        
        # 5. Removing wind cells
        noWind = partBirthScalefac > 0
        
        partPosCo_noWind            = partPosCo[noWind]   
        partPosPr_noWind            = partPosPr[noWind]
        partMass_noWind             = partMass[noWind]
        partBirthScalefac_noWind    = partBirthScalefac[noWind]
        partBirthMetallicity_nowind = partBirthMetallicity[noWind]
        partBirthMass_nowind        = partBirthMass[noWind]
        
        """Calculating the age of all stellar particles (time between birth and snapshot)"""
        """Creating a mask to separate stellar particles with age < 30 Myr (young) and age > 30 Myr (old)"""
        """If no stellar particles then returns empty age array"""
        ageThreshold = 30 * u.Myr
        if len(partBirthScalefac_noWind) == 0:
            partRedshift_noWind   = np.array([])
            partCosmicTime_nowind = np.array([])
            partAge_noWind        = np.array([]) * u.Gyr
    
        else:
            partRedshift_noWind   = (1 / partBirthScalefac_noWind) - 1
            partCosmicTime_noWind = self.cosmo.age(partRedshift_noWind)              # AstroPy leaves age in Gyr
            partAge_noWind        = self.cosmo.age(redshift) - partCosmicTime_noWind # AstroPy leaves age in Gyr
            
        young_mask = partAge_noWind.to(u.Gyr) < ageThreshold.to(u.Gyr) if len(partAge_noWind) > 0 else np.array([], dtype = bool)
    
        # 6. Helper for Dictionary Construction
        def get_part_dict(mask):
            indices = np.where(mask)[0]
            return {
                'FormTime':         partCosmicTime_noWind[indices],
                'PosPr':            partPosPr_noWind[indices],
                'PosCo':            partPosCo_noWind[indices],
                'Mass':             partMass_noWind[indices],
                'BirthMass':        partBirthMass_nowind[indices],
                'BirthMetallicity': partBirthMetallicity_nowind[indices],
                'Age':              partAge_noWind[indices],
                'Count':            np.sum(mask)
            }
    
        return {
            'particles': {
                'young': get_part_dict(young_mask),
                'old':   get_part_dict(~young_mask)
            },
            'subhalo': {
                'PosPr':             subPosPr,
                'PosCo':             subPosCo,
                'SFRIn2HfStellRad':  subSfrIn2HfStellRad,
                'Zgas':              subGasMetallicity,
                'StellHfMassRadPr':  subHfStellMassRadPr,
                'StellHfMassRadCo':  subHfStellMassRadCo,
                'StellMassIn2HfRad': subStellMassIn2HfStellRad,
                'TotStellMass': subTotStellMass,
                'GroupNum': GrNr,
                'isCentral': isCentral
            },
    
            'fof':{
                'm200': fofM200,
                'r200_pr': fofR200Pr,
                'r200_co': fofR200Co            
            },
            
            'Idx': idx
        }

    def compute_all_apertures(self, simName, snapNum, R=30*u.kpc, ncpu=1, save_data = False, overwrite = False):
        """Method that computes all subhalo apertures with radius R using multiprocessing"""

        fname = f'apDict_{simName}_snap{snapNum}'
        path = os.path.join(self.pickle_dir, fname)
        apDict_exists = os.path.exists(path)

        if apDict_exists and not overwrite:
            print(f"Aperture dictionary already exists for {simName} at snapshot {snapNum}!")
            print(f"Loading aperture dictionary from {path}...")
            with open(path, 'rb') as handle:
                apDict = pickle.load(handle)

            print("Successfully loaded aperture dictionary!")

            return apDict
            
        else:
            
            data_loaded = (
                simName in self.stellPartsDict
                and snapNum in self.stellPartsDict[simName]
                and simName in self.subhaloDict
                and snapNum in self.subhaloDict[simName]
                and simName in self.simDict
                and snapNum in self.simDict[simName]
                and simName in self.fofDict
                and snapNum in self.fofDict[simName]
            )
    
            if not data_loaded:
                self.load_data()
    
            # Find valid subhalos
            valid_idxs = self.identify_subhalo_idxs(simName=simName, snapNum=snapNum)
    
            print("TOTAL SUBHALOS:",
                  len(self.subhaloDict[simName][snapNum]['SubhaloLenType']))
            
            print("VALID SUBHALOS:",
                  len(valid_idxs))
    
            # Build KDTree 
            if (
                self.starTree is None
                or self.tree_simName != simName
                or self.tree_snapNum != snapNum
            ):
                self.build_star_tree(simName, snapNum)
    
            # Arguments for each calculation
            args = [(idx, snapNum, simName, R) for idx in valid_idxs]
    
            # Multiprocessing
            with mp.Pool(processes=ncpu, initializer=init_worker, initargs=(self,)) as pool:
                results = list(tqdm.tqdm(pool.imap(compute_aperture_worker, args), total=len(args)))
    
            # Convert to dictionary indexed by subhalo index
            results_dict = {
                result['Idx']: result
                for result in results
            }

            if save_data:
                with open(path, 'wb') as handle:
                    pickle.dump(results_dict, handle)
                print(f"Successfully saved aperture dictionary to {path}")
                
            return results_dict     