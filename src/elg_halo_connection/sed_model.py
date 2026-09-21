import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import tqdm
import os

# Load FSPS
environdir = "/cosma8/data/dp004/dc-gilf1/projects/MG_project/fsps"
os.environ["SPS_HOME"] = environdir
import fsps



class GalaxySEDModel:
    def __init__(self, numGals, headersDict, cal_res_dict, DESI_flux_df, minStellPart, bands = ['des_g', 'des_r', 'des_z'], dust_index = -0.7, compute_vega_mags = False, ionParam = -1.4):
        """
        Contains methods to calculate the colour magnitudes of the flux from both old and young stars for subhalos in IllustrisTNG simulations
        pOld we have zcontinuous = 3 and sfh = 3 since we create our own SFH table for the old particles
        spYoung we have add_neb_emission = True and add_neb_continuum = True since young stars ionize surrounding gas. FSPS requires zcontinous = 1 when enabling nebular emission
        
        """
        
        # Aperture dictionary with optical depth added for each subhalo
        self.cal_res_dict      = cal_res_dict
        self.apDict            = self.cal_res_dict['Aperture_Snapshot']['Dict']
        self.DESI_flux_df      = DESI_flux_df
        

        self.redshift          = headersDict['Redshift']
        self.h                 = headersDict['HubbleParam']
        self.ageSnapshot       = cosmo.age(headersDict['Redshift'])
        self.Z_sun             = 0.014 # Consistent with the MIST isochrones

        self.dust_index        = dust_index
        self.ionParam          = ionParam
        self.compute_vega_mags = compute_vega_mags
        self.bands             = bands
        self.dust_tesc         = np.log10(3e7) # 30 Myr threshold consistent with Hadzhiyska 2021
        
        # Only select subhalos with non-zero gas-phase metallicity and with a minimum stellar particle count
        self.headersDict    = headersDict
        self.minStellPart   = minStellPart
        self.filteredData   = [(idx, data) for idx, data in self.apDict.items() if (data['subhalo']['Zgas'] > 0) and ((data['particles']['old']['Count'] + data['particles']['young']['Count']) >= minStellPart)]

        if numGals == None:
            self.numGals = len(self.filteredData)

        else:
            self.numGals = min(numGals, len(self.filteredData))

        self.g_5sigma = 24.0
        self.r_5sigma = 23.4
        self.z_5sigma = 22.5

        self.m_5sigma = np.array([self.g_5sigma, self.r_5sigma, self.z_5sigma])

    def compute_fluxDense_old(self, nAgeBins = 40):
        """Method to compute the flux from stars with age > 30 Myr
           Input:
               - nAgeBins: Number of form time (cosmic time) bins for the SFH table for the old particles

           Output:
               - fluxDenseArrOld: len(filteredData) x len(bands) array containing total old flux in each band for each subhalo
        """
        spOld   = fsps.StellarPopulation(
                                  compute_vega_mags = self.compute_vega_mags, 
                                  zcontinuous = 3, 
                                  sfh = 3, 
                                  dust_index = self.dust_index, 
                                  zred = self.redshift, 
                                  dust_tesc = self.dust_tesc, 
                                  dust1 = 0, 
                                  add_neb_continuum = False
        )
        fluxDenseListOld = []
        haloIdxListOld   = []

        # Define age bins for SFH table
        minAge, maxAge = 0 * u.Gyr, self.ageSnapshot
        ageBins = np.linspace(minAge, maxAge, nAgeBins + 1)
        dt = ageBins[1] - ageBins[0]
        
        for i in tqdm.tqdm(range(self.numGals), desc = "Computing Old Fluxes"):
            haloIdx, haloDict = self.filteredData[i]
            haloIdxListOld.append(haloIdx)

            # Model 
            tauV = haloDict['subhalo']['tau_v']
            spOld.params['dust2'] = tauV
        
            oldPartDict = haloDict['particles']['old']
        
            oldPartStellMasses    = oldPartDict['BirthMass']        # Msun
            oldPartStellFormTimes = oldPartDict['FormTime']         # Gyr (cosmic time)
            oldPartStellAges      = oldPartDict['Age']              # Gyr
            oldPartStellMets      = oldPartDict['BirthMetallicity'] # Absolute

            min_old_age = 10 ** self.dust_tesc * u.yr
            assert np.all(oldPartStellAges >= min_old_age), "Error: Particles younger than 30 Myr found in old component!"
            
            binIndices      = np.digitize(oldPartStellFormTimes, ageBins) - 1
            nIntervals      = len(ageBins) - 1
            valid           = (binIndices >= 0) & (binIndices < nIntervals)
            
            validBinIndices = binIndices[valid]
            validMasses     = oldPartStellMasses[valid]
            validMets       = oldPartStellMets[valid]
            
            totBinMasses    = np.bincount(validBinIndices, weights = validMasses, minlength = nIntervals)
            weightedBinMets = np.bincount(validBinIndices, weights = validMasses * validMets, minlength = nIntervals)
            
            binSFR = (totBinMasses / dt).to(u.Msun / u.yr)
            binMet = np.where(totBinMasses.to_value(u.Msun) > 0, (weightedBinMets / totBinMasses).to_value(u.dimensionless_unscaled), 1e-4)
            binAge = 0.5 * (ageBins[:-1] + ageBins[1:])
            
            if len(binSFR) != len(binMet):
                raise Exception("binSFR and binMet lengths do not match!")
            
            fspsAge    = binAge.to_value(u.Gyr)
            fspsSFR    = binSFR.to_value(u.Msun / u.yr)
            fspsMet    = binMet
            
            spOld.set_tabular_sfh(age = fspsAge, sfr = fspsSFR, Z = fspsMet)
            
            oldMags = spOld.get_mags(tage = self.ageSnapshot.to_value(u.Gyr), bands = self.bands)
                
            fluxDense = 10 ** (-0.4 * oldMags)
        
            fluxDenseListOld.append(fluxDense)
        
        fluxDenseArrOld = np.array(fluxDenseListOld)
        haloIdxArrOld   = np.array(haloIdxListOld)
    
        return haloIdxArrOld, fluxDenseArrOld

    def compute_fluxDense_young(self, dust1 = 0.3):
        """Method to compute the flux from stars with age < 30 Myr
           Input:
               - dust1: Scalar quantity which is the optical depth in the V-band of the birth cloud

           Output:
               - fluxDenseArrYoung: len(filteredData) x len(bands) array containing total young flux in each band for each subhalo
        
        """

        spYoung = fsps.StellarPopulation(compute_vega_mags = self.compute_vega_mags, 
                                  zcontinuous = 1, 
                                  add_neb_emission = True, 
                                  add_neb_continuum = True, 
                                  gas_logu = self.ionParam, 
                                  dust_index = self.dust_index,
                                  zred = self.redshift,
                                  dust_tesc = self.dust_tesc)

            
        fluxDenseListYoung = []
        haloIdxListYoung   = []
        
        for i in tqdm.tqdm(range(self.numGals), desc = "Computing Young Fluxes"):
            haloIdx, haloDict = self.filteredData[i]
            haloIdxListYoung.append(haloIdx)
        
            tauV = haloDict['subhalo']['tau_v']
            
            spYoung.params['dust1'] = dust1
            spYoung.params['dust2'] = tauV
        
            youngPartDict = haloDict['particles']['young']
            
            youngPartStellMasses = youngPartDict['BirthMass']
            youngPartStellAges   = youngPartDict['Age'] 
            youngPartStellMets   = youngPartDict['BirthMetallicity']

            # Young flux contribution to g, r, z band for this subhalo (1 x 3 array)
            fluxDense = np.zeros_like(self.bands, dtype = float)

            if len(youngPartStellMasses) == 0:
                fluxDenseListYoung.append(fluxDense)
                continue
                  
            # Ensure that all particles have age < 30 Myr
            max_young_age = 10 ** self.dust_tesc * u.yr
            assert np.all(youngPartStellAges <= max_young_age), "Error: Particles older than 30 Myr found in young component!"

            youngPartStellMasses_val = youngPartDict['BirthMass'].to_value(u.Msun)
            youngPartStellAges_val   = youngPartDict['Age'].to_value(u.Gyr)
            youngPartStellMets_val   = youngPartDict['BirthMetallicity']

            gasPhaseMetallicity = haloDict['subhalo']['Zgas']


            # Loop through each young particle and execute spYoung.get_mags
            # Set gas-phase metallicity to halo 
            spYoung.params['gas_logz'] = np.log10(gasPhaseMetallicity / self.Z_sun)

            for mass_val, age_val, Zstar_val in zip(youngPartStellMasses_val, youngPartStellAges_val, youngPartStellMets_val):
                spYoung.params['logzsol'] = np.log10(Zstar_val / self.Z_sun)
                age_val = max(age_val, 1e-4)
                youngMags = spYoung.get_mags(tage = age_val, bands = self.bands)
                fluxDense += mass_val * (10**(-0.4 * youngMags))
            
            fluxDenseListYoung.append(fluxDense)
    
        fluxDenseArrYoung = np.array(fluxDenseListYoung)
        haloIdxArrYoung   = np.array(haloIdxListYoung)
    
        return haloIdxArrYoung, fluxDenseArrYoung

    def apply_desi_mask(self, g_arr, r_arr, z_arr):
        """Applies the DESI magnitude cuts and limits to the raw colour arrays"""

        DESI_mask = (g_arr < 23.6) & \
                    (g_arr > 20.0) & \
                    (r_arr < self.r_5sigma) & \
                    (z_arr < self.z_5sigma)

        g_DESI, r_DESI, z_DESI = g_arr[DESI_mask], r_arr[DESI_mask], z_arr[DESI_mask]

        return g_DESI, r_DESI, z_DESI, DESI_mask
        

    def compute_mags(self, dust1 = 0.3, numAgeBins = 40, delta_m = 0):
        """Method to compute the r-z and g-r flux arrays for the simulated subhalos by adding photometric noise to the old and young flux
           Input:
               - dust1: Optical depth for stellar birth clouds
               - numAgeBins: Number of bins for old-particle form times (cosmic time) for SFH bin
               - delta_m: Optional correction factor

           Output:
               - g_r_array: g-r colour magnitude array
               - r_z_array: r-z colour magnitude array
               - validHaloIdxsArr: array containing snapshot indices of halos within the initial magnitude cuts
        
        """
        haloIdxArrYoung, fluxDenseArrYoung = self.compute_fluxDense_young(dust1)
        print("Successfully computed young SSP fluxes!")
        
        haloIdxArrOld, fluxDenseArrOld   = self.compute_fluxDense_old(numAgeBins)
        print("Successfully computed old SSP fluxes!")

        assert np.array_equal(haloIdxArrYoung, haloIdxArrOld), "Young and old halo indices are not aligned"

        # Both haloIdxArrOld and haloIdxArrYoung are equal so just create new variable
        haloIdxArrRaw = haloIdxArrOld

        totFluxDense = fluxDenseArrOld + fluxDenseArrYoung
    
        # Apply Noise
        flux_5sigma = 10 ** (-0.4 * self.m_5sigma)
        flux_sigma  = flux_5sigma / 5
    
        fluxNoise    = np.random.normal(loc = 0.0, scale = flux_sigma, size = totFluxDense.shape)
        noisyFlux = totFluxDense + fluxNoise
    
        noisyMags = np.full_like(noisyFlux, np.nan, dtype=float)
        isValid   = noisyFlux > 0
    
        noisyMags[isValid] = -2.5 * np.log10(noisyFlux[isValid]) + (np.ones_like(noisyMags[isValid]) * delta_m)

        noisy_g, noisy_r, noisy_z = noisyMags[:, 0], noisyMags[:, 1], noisyMags[:, 2]

        noisy_g_DESI, noisy_r_DESI, noisy_z_DESI, DESI_mask = self.apply_desi_mask(noisy_g, noisy_r, noisy_z)

        g_r_array = noisy_g_DESI - noisy_r_DESI
        r_z_array = noisy_r_DESI - noisy_z_DESI

        validHaloIdxsArr   = haloIdxArrRaw[DESI_mask]

        assert len(validHaloIdxsArr) == len(g_r_array), "DESI magnitude cut halo IDs and flux data are mismatched"
        
        return g_r_array, r_z_array, validHaloIdxsArr

    def apply_elg_cuts(self, g_r_arr, r_z_arr):
        """Apply ELG selection cuts by DESI Collaboration 2016 to the raw r-z and g-r arrays
           Input: 
               - g-r array
               - r-z array
    
          Output:
              - g-r array with ELG cut mask applied
              - r-z array with ELG cut mask applied
        
        """
    
        ELG_mask = (r_z_arr < 1.6) & (r_z_arr > 0.3) & (g_r_arr < 1.15 * r_z_arr - 0.15) & (g_r_arr < -1.2 * r_z_arr + 1.6)
    
        g_r_arr_masked, r_z_arr_masked = g_r_arr[ELG_mask], r_z_arr[ELG_mask]
    
        return g_r_arr_masked, r_z_arr_masked, ELG_mask

    def compute_num_density(self, num_galaxies):

        boxsize_kpc = self.headersDict['BoxSize'] * u.kpc
        boxsize_Mpc = boxsize_kpc.to(u.Mpc)

        volume = boxsize_Mpc**3
    
        numDensity = num_galaxies / volume
    
        return numDensity

    def get_desi_subhalos(self):
        
        flux_df = self.DESI_flux_df
        flux_g, flux_r, flux_z = flux_df['flux_g'], flux_df['flux_r'], flux_df['flux_z']
        
        def comp_desi_mags(flux):
            """Consistent with https://www.legacysurvey.org/dr8/description/"""
            mags = 22.5 - 2.5 * np.log10(flux)
            return mags
        
        mag_g, mag_r, mag_z = comp_desi_mags(flux_g), comp_desi_mags(flux_r), comp_desi_mags(flux_z)

        mag_g_DESI, mag_r_DESI, mag_z_DESI, mask_DESI = self.apply_desi_mask(mag_g, mag_r, mag_z) 
        
        mag_g_r_DESI = mag_g_DESI - mag_r_DESI
        mag_r_z_DESI = mag_r_DESI - mag_z_DESI
        
        obsRedshifts = flux_df['redshift']
        redshift_DESI  = np.mean(obsRedshifts[mask_DESI])

        return mag_g_r_DESI, mag_r_z_DESI, redshift_DESI

    def get_magDict(self, dust1, numAgeBins, delta_m = 0.0):
        """Return dictionary containing cut and uncut magnitudes and the number density of subhalos for each case
           Input:
               - dust1: Optical depth for birth clouds
               - numAgeBins: Number of bins for old-particle form times (cosmic time) for SFH bin
               - delta_m: Optional correction factor for colour magnitudes

           Output:
               - magDict: Dictionary containing number density, r-z and g-r arrays for both ELG cuts and uncuts
               - ELG_HaloData: List containing dictionary for each ELG
        """

        # Simulation
        g_r_uncut_sim, r_z_uncut_sim, validHaloIdxsArr_sim = self.compute_mags(dust1, numAgeBins, delta_m)
        g_r_cut_sim, r_z_cut_sim, ELG_mask_sim             = self.apply_elg_cuts(g_r_uncut_sim, r_z_uncut_sim)

        haloIdxsArr_ELG = validHaloIdxsArr_sim[ELG_mask_sim]

        assert len(haloIdxsArr_ELG) == len(g_r_cut_sim), "Mismatch between ELG cut magnitude array and ELG cut halo indices"

        haloDataDict = {haloIdx: data for haloIdx, data in self.filteredData}
        ELG_HaloData = [haloDataDict[haloIdx] for haloIdx in haloIdxsArr_ELG]

        numDensity_cut_sim   = self.compute_num_density(len(g_r_cut_sim))
        numDensity_uncut_sim = self.compute_num_density(len(g_r_uncut_sim))

        # DESI survey
        g_r_uncut_DESI, r_z_uncut_DESI, _ = self.get_desi_subhalos()
        g_r_cut_DESI, r_z_cut_DESI, ELG_mask_DESI     = self.apply_elg_cuts(g_r_uncut_DESI, r_z_uncut_DESI)

        numDensity_cut_DESI    = None
        numDensity_uncut_DESI  = None

        magDict = {
            'Sim': {
                'cut': {
                    'numDensity': numDensity_cut_sim,
                    'r-z': r_z_cut_sim,
                    'g-r': g_r_cut_sim,
                },
                'uncut': {
                    'numDensity': numDensity_uncut_sim,
                    'r-z': r_z_uncut_sim,
                    'g-r': g_r_uncut_sim,
                },
                'ELG_HaloData': ELG_HaloData
            },
            'DESI': {
                'cut': {
                    'numDensity': numDensity_cut_DESI,
                    'r-z': r_z_cut_DESI,
                    'g-r': g_r_cut_DESI,
                },
                'uncut': {
                    'numDensity': numDensity_uncut_DESI,
                    'r-z': r_z_uncut_DESI,
                    'g-r': g_r_uncut_DESI,
                },
            },
        }
                   

        return magDict
