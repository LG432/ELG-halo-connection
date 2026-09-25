import multiprocessing as mp
import astropy.units as u
import numpy as np
import tqdm
from scipy.spatial import cKDTree

_worker_environment = None

def init_environment_worker(environment):
    global _worker_environment
    _worker_environment = environment

def environment_worker(idx):
    return _worker_environment.compute(idx)


class EnvironmentCalculator:
    def __init__(self, subhaloDict, fofDict, simDict, R_min = 0.5 * u.Mpc, R_max = 4 * u.Mpc, min_DM_mass = 1e9 * u.Msun):
        self.subhaloDict = subhaloDict
        self.fofDict     = fofDict
        self.simDict     = simDict
        
        self.R_min       = R_min
        self.R_max       = R_max
        self.min_DM_mass = min_DM_mass

        self.dmTree      = None
        self.simName     = None
        self.snapNum     = None

    def build_tree(self, snapNum = None, simName = None):
        """Build tree for dark matter halo positions in units pkpc"""

        self.simName = simName
        self.snapNum = snapNum

        sim = self.simDict[simName][snapNum]
        sub = self.subhaloDict[simName][snapNum]
    
        h          = sim['HubbleParam']
        redshift   = sim['Redshift']
        boxsize_co = sim['BoxSize'] * u.kpc / h
    
        scaleFac   = 1 / (1 + redshift)
        boxsize_pr = boxsize_co * scaleFac
    
        # Position of all subhalos within the simulation volume
        subhalo_pos_co = sub['SubhaloPos'] * u.kpc / h
        subhalo_pos_pr = subhalo_pos_co * scaleFac
    
        # KD Tree of all subhalo positions in the simulation volume
        self.dmTree = cKDTree(subhalo_pos_pr.to_value(u.kpc), boxsize_pr.to_value(u.kpc))   
        
    def get_properties(self, idx, simName = None, snapNum = None):
        """Extract relevant subhalo, headers and fof properties from their dictionaries"""

        if simName is None:
            simName = self.simName

        if snapNum is None:
            snapNum = self.snapNum
        
        props_sub  = self.subhaloDict[simName][snapNum] # Subhalo properties
        props_fof  = self.fofDict[simName][snapNum]     # FoF group properties
        props_sim  = self.simDict[simName][snapNum]     # Simulation properties
    
        #------------- Simulation Properties -------------#
        
        # Hubble parameter
        h        = props_sim['HubbleParam'] 
        redshift = props_sim['Redshift']    
        scaleFac = 1 / (1 + redshift) 

        # Simulation boxsize
        boxsize_co_h = props_sim['BoxSize'] * u.kpc # ckpc h^-1    
        boxsize_co   = boxsize_co_h / h             # ckpc
        boxsize_pr   = boxsize_co * scaleFac        # pkpc
    
        # Simulation volume
        volume_pr   = boxsize_pr**3   # pMpc^3
        
        #------------- Subhalo Properties -------------#
    
        # Subhalo total dark matter mass
        all_sub_TotDarkMass_h = props_sub['SubhaloMassType'][:, 1] * 1e10 * u.Msun # Msun
        all_sub_TotDarkMass   = all_sub_TotDarkMass_h / h
    
    
        # Subhalo position of particle with minimum gravitational potential energy
        sub_pos_co_h = props_sub['SubhaloPos'][idx] * u.kpc
        sub_pos_co   = sub_pos_co_h / h
        sub_pos_pr   = sub_pos_co * scaleFac
    
        # FoF group number of subhalo
        sub_GrNr = props_sub['SubhaloGrNr'][idx]
    
        #------------- FoF Properties -------------#
        
        GrNr = sub_GrNr
        
        fof_pos_co_h = props_fof['GroupPos'][GrNr] * u.kpc
        fof_pos_co   = fof_pos_co_h / h
        fof_pos_pr   = fof_pos_co * scaleFac
    
        fof_m200_h   = props_fof['Group_M_Crit200'][GrNr] * 1e10 * u.Msun
        fof_m200     = fof_m200_h / h
    
        fof_r200_h   = props_fof['Group_R_Crit200'][GrNr] * u.kpc
        fof_r200     = fof_r200_h / h
    
        return {
            'Simulation': {
                'HubbleParam': h,
                'Redshift': redshift,
                'ScaleFactor': scaleFac,
                'BoxSize': boxsize_pr,
                'Volume': volume_pr
                
            },
            'Subhalo': {
                'Idx': idx,
                'GrNr': GrNr,
                'All_DM_Mass': all_sub_TotDarkMass,
                'Position': sub_pos_pr                
            },
            'FoF': {
                'M200': fof_m200,
                'R200': fof_r200,
                'Position': fof_pos_pr
            }
               }

    def calculate_rel_dense(self, position, all_DM_mass, volume, R_min, R_max):
        """Calculates relative envrionment density for both subhalos and FoFs"""
    
        # Query all subhalos in range R200 < R < R_max
        R_idxs_min = self.dmTree.query_ball_point(position.to_value(u.kpc), R_min.to_value(u.kpc))
        R_idxs_max = self.dmTree.query_ball_point(position.to_value(u.kpc), R_max.to_value(u.kpc))
    
        # Remove subhalos within R < R_min
        R_idxs = list(set(R_idxs_max) - set(R_idxs_min))
        
        # Select only subhaloes within aperture range
        in_range_masses = all_DM_mass[R_idxs]
        
        selected_masses = in_range_masses[in_range_masses > self.min_DM_mass]

        # Sum of all subhalo DM masses
        selected_total_mass = np.sum(selected_masses)
            
        # Volume of R200 < R < R_max
        shell_volume = (4/3) * np.pi * (R_max**3 - R_min**3)
    
        # DM density of all subhalos within the annulus and above the minimum DM mass
        selected_density = selected_total_mass / shell_volume

        valid_masses = all_DM_mass[all_DM_mass > self.min_DM_mass]
        mean_density = np.sum(valid_masses) / volume
    
        # Relative DM density of all subhalos within the annulus and above the minimum DM mass
        relative_density = (selected_density / mean_density).to_value(u.dimensionless_unscaled)

        return relative_density
        
    def compute(self, idx, simName = None, snapNum = None):

        if self.dmTree is None:
            raise RuntimeError("KDTree has not been built. Run build_tree() first.")

        if simName is None:
            simName = self.simName

        if snapNum is None:
            snapNum = self.snapNum

        props = self.get_properties(idx, simName, snapNum)

        sub = props['Subhalo']
        sim = props['Simulation']
        fof = props['FoF']

        position = sub['Position']

        #------------- Nearest Neighbours -------------#
    
        # Query the 11 nearest subhalos.
        # The first entry is the subhalo itself, at distance = 0,
        # so the remaining entries correspond to the 1st, 2nd, ... nearest neighbours.    
        nearest_distances, _ = self.dmTree.query(position.to_value(u.kpc), k=11)
    
        # Exclude the subhalo itself (first entry)
        nearest_distances = nearest_distances[1:]
    
        # Distances of the 5th, 8th and 10th nearest neighbours
        fifth_nearest_distance  = nearest_distances[4] * u.kpc
        eighth_nearest_distance = nearest_distances[7] * u.kpc
        tenth_nearest_distance  = nearest_distances[9] * u.kpc
        
        #------------- Subhalo Environment -------------#

        relative_density_subhalo = self.calculate_rel_dense(
            position = position, 
            all_DM_mass = sub['All_DM_Mass'], 
            volume = sim['Volume'], 
            R_min = self.R_min, 
            R_max = self.R_max
        )

        #------------- FoF Environment -------------# 

        relative_density_fof = self.calculate_rel_dense(
            position = fof['Position'],
            all_DM_mass = sub['All_DM_Mass'],
            volume = sim['Volume'],
            R_min = fof['R200'],
            R_max = self.R_max
                                                       
        )
                         
        res = {
            'Subhalo_Idx': idx,

            'FoF_GroupNumber': sub['GrNr'],

            'Relative_Density': {
                'Subhalo': relative_density_subhalo,
                'FoF': relative_density_fof
            },

            'Nearest_Subhalo_Distance': {
                '5th': fifth_nearest_distance,
                '8th': eighth_nearest_distance,
                '10th': tenth_nearest_distance
            }
        }

        return {simName:
               {
                   snapNum
               }
               }

    def compute_all_subhalos(self, subhalo_idxs, ncpu=1):

        if self.dmTree is None:
            raise RuntimeError("KDTree has not been built. Run build_tree() first.")

        with mp.Pool(processes=ncpu, initializer=init_environment_worker, initargs = (self, )) as pool:
            results = list(tqdm.tqdm(pool.imap(environment_worker, subhalo_idxs), total = len(subhalo_idxs), desc = "Calculating environment"))

        return {result['Subhalo_Idx']: result for result in results}
