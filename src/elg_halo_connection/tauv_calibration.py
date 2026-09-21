import numpy as np
import astropy.units as u
from scipy.stats import ks_2samp
from scipy.optimize import minimize

class DustAttenuationCalibrator:
    def __init__(self, headersDict_cal, subhaloDict_cal, df_GSWLC, minStellPart=100):
        """
        Key variables:
            df_GSWLC                 - GALEX-SDSS-WISE LEGACY CATALOG dataframe (https://salims.pages.iu.edu/gswlc/) (z ~ 0.1)
            headersDict_cal          - dictionary containing the simulation headers at the GSWLC redshift 
            subhaloDict_cal          - dictionary containing the subhalo attributes at the GSWLC redshift
            minStellPart             - minimum stellar particles for a simulation subhalo to be considered resolved
            Z_sun                    - solar metallicity quoted in TNG300-1
            gasPhaseMetallicity      - Mass weighted average metallicities of all gas cells in each subhalo
            numPart_Metallicity_mask - Mask to exclude subhalos with zero gas-phase metallicity and too few stellar particles
        
        """
        
        self.headersDict_cal    = headersDict_cal
        self.subhaloDict_cal    = subhaloDict_cal
        self.df_GSWLC           = df_GSWLC
        self.minStellPart       = minStellPart

        self.redshift       = headersDict_cal['Redshift']
        self.h              = headersDict_cal['HubbleParam']
        self.Z_sun          = 0.0127

        self.numStellPart             = subhaloDict_cal['SubhaloLenType'][:, 4]
        self.gasPhaseMetallicity      = subhaloDict_cal['SubhaloGasMetallicity']
        self.numPart_Metallicity_mask = (self.numStellPart >= minStellPart) & (self.gasPhaseMetallicity > 0) 

    def compute_tauV(self, totStellMass, hfStellMassRad, gasPhaseMetallicity, alpha, beta, gamma):
        """
        Function: 
            Computes the V-band optical depths for the ISM of each subhalo using Eq.(1) from Hadzhiyska 2021.

        Key variables:
            totStellMass        - total stellar mass of all bound star particles to each subhalo
            hfStellMassRad      - radius enclosing half the stellar mass to each subhalo
            gasPhaseMetallicity - Mass weighted average metallicities of all gas cells in each subhalo
            alpha, beta, gamma  - Parameters

        Output:
            Array containing the V-band optical depth for each subhalo
        
        """
        SigmaArr   = totStellMass / (np.pi * hfStellMassRad**2)
        meanSigma  = np.mean(SigmaArr)
        SigmaTilde = SigmaArr / meanSigma

        tauV_arr = (gamma * ((gasPhaseMetallicity / self.Z_sun)**alpha) * (SigmaTilde ** beta)).to_value(u.dimensionless_unscaled)

        return tauV_arr

    
    def compute_tauV_from_subDict(self, alpha, beta, gamma):
        """
        Function:
            Takes the data stored within simulationDict_cal, subhaloDict_cal to calculate V-band optical depths for given alpha, beta, gamma
            Uses compute_tauV to do this computation

        Key Variables:
            mask                       - gas-phase metallicity > 0 and number star particles > minStellPart
            hfStellMassRad_masked      - half stellar mass radius for each subhalo within the mask
            totStellMass_masked        - total stellar mass for each subhalo within the mask
            gasPhaseMetallicity_masked - Mass-weighted metallicity for each subhalo within the mask

        Output:
            Array containing V-band optical depth for each subhalo
            Same output as compute_tauV
            
        """

        mask = self.numPart_Metallicity_mask

        scaleFac = 1 / (1 + self.redshift)
    
        hfStellMassRad_masked      = self.subhaloDict_cal['SubhaloHalfmassRadType'][:, 4][mask] * scaleFac * u.kpc / self.h
        totStellMass_masked        = self.subhaloDict_cal['SubhaloMassInRadType'][:, 4][mask] * 1e10 * u.Msun / self.h
        gasPhaseMetallicity_masked = self.subhaloDict_cal['SubhaloGasMetallicity'][mask]
        
        return self.compute_tauV(totStellMass_masked, hfStellMassRad_masked, gasPhaseMetallicity_masked, alpha, beta, gamma)
    
    def compute_GSWLC_tauV(self):
        """
        Function:
            Preps the GSWLC catalogue: 
                - accounts for the minimum simulation halo stellar mass depending on minStellPart
                - neglects tauV <= 0 since we cannot achieve these values with the empirical expression if alpha < 0

        Key Variables:
            minMass    - minimum stellar mass of all the simulated subhalos within the mask
            mask_GSWLC - excludes galaxies with tauV <= 0 and stellar masses below the minimum simulated subhalo stellar mass

        Output:
            df_GSWLC_valid - cleaned GSWLC dataframe
            GSWLC_tauV     - V-band optical depths of galaxies within the cleaned GSWLC dataframe
             
        """       

        mask = self.numPart_Metallicity_mask
        
        totStellMass_masked = self.subhaloDict_cal['SubhaloMassInRadType'][:, 4][mask] * 1e10 * u.Msun / self.h
        
        minMass       = np.min(totStellMass_masked)
        log10_minMass = np.log10(minMass.to_value(u.Msun))
    
        mask_GSWLC = (self.df_GSWLC['tau_v'] > 0) & (self.df_GSWLC['log10(M*)'] > log10_minMass)
        
        df_GSWLC_valid  = self.df_GSWLC[mask_GSWLC]
        
        GSWLC_tauV  = df_GSWLC_valid['tau_v']
    
        return df_GSWLC_valid, GSWLC_tauV
    
    
    def ks_objective_func(self, params, obsTauV, alpha):
        """
        Function:
            Calculates KS statistic for simulated V-band optical depth PDF by comparing to GSWLC V-band optical depth PDF

        Key Variables:
            alpha               - Preset parameter (-0.6 according to Hadzhiyska 2021)
            beta, gamma         - Parameters for V-band optical depth expression
            sim_tauV            - V-band optical depths for simulated subhalos computed using compute_tauV_from_subDict
            ks_result           - Object containing KS statistic attributes
            ks_result.statistic - KS statistic value
            
        """
        beta, gamma = params
    
        sim_tauV  = self.compute_tauV_from_subDict(alpha, beta, gamma)
        ks_result = ks_2samp(sim_tauV, obsTauV)
    
        return ks_result.statistic
    
    def compute_optimal_params(self, initialGuess, alpha, maxiter):
        """
        Function:
            Performs minimization of the KS statistic by varying beta and gamma
            Minimum KS statistic gives optimal beta and optimal gamma (optBeta, optGamma)

        Key Variables:
            initialGuess - tuple containing the initial guess for beta, gamma (beta, gamma)
            method       - minimization method
            maxiter      - maximum number of iterations before giving statistic

        Output:
            Tuple containing the beta and gamma values that minimise the
            two-sample KS statistic between the simulated and GSWLC tau_V
            distributions.
                
        
        """
    
        df_GSWLC_valid, GSWLC_tauV = self.compute_GSWLC_tauV()
    
        res = minimize(self.ks_objective_func, 
                       initialGuess, 
                       args=(GSWLC_tauV, alpha), 
                       method = 'Nelder-Mead', 
                       options = {'maxiter': maxiter})
    
        optBeta, optGamma = res.x[0], res.x[1]
    
        return optBeta, optGamma

    def compute_tauV_from_apDict(self, apDict, alpha, beta, gamma):
        """
        Function:
            Computes the V-band optical depths for all subhalos within apDict using optimal alpha, beta, gamma

        Key Variables:
            filteredData               - array containing dictionaries for all subhalos from apDict with gas-phase metallicity > 0
            hfStellMassRadPr_masked    - half stellar mass radius of subhalos in filteredData
            totStellMass_masked        - total stellar mass of subhalos in filteredData
            gasPhaseMetallicity_masked - mass weighted average gas-phase metallicity of subhalos in filteredData

        Output:
            Array containing V-band optical depths of all subhalos from apDict
            Uses compute_tauV for this computation
        
        
        """


        filteredData = [data for idx, data in apDict.items() if data['subhalo']['Zgas'] > 0]
        
        hfStellMassRadPr_masked    = u.Quantity([d['subhalo']['StellHfMassRadPr'] for d in filteredData])
        totStellMass_masked        = u.Quantity([d['subhalo']['StellMassIn2HfRad'] for d in filteredData])
        gasPhaseMetallicity_masked = u.Quantity([d['subhalo']['Zgas'] for d in filteredData])
    
        return self.compute_tauV(totStellMass_masked, hfStellMassRadPr_masked, gasPhaseMetallicity_masked, alpha, beta, gamma)

        
    def assign_optTauV(self, apDict, alpha, optBeta, optGamma):
        """
        Function:
            Assigns the V-band optical depths from compute_tauV_from_apDict into the original aperture dictionary apDict

        Key Variables:
            optBeta, optGamma - Values of beta and gamma that minimise the two-sample KS statistic between the simulated and GSWLC tauV distributions
            opt_tauV          - tauV values obtained by use of optBeta and optGamma

        Output:
            Original aperture dictionary apDict with the addition of the 'tau_v' field for each subhalo
            Subhalos that were masked out before get a nan value for 'tau_v'
        
        
        """

        opt_tauV          = self.compute_tauV_from_apDict(apDict, alpha, optBeta, optGamma)

        # Assign opt_tauV to subhaloDict
        filteredCounter = 0
        
        for idx, data in apDict.items():
            if data['subhalo']['Zgas'] > 0:
                apDict[idx]['subhalo']['tau_v'] = opt_tauV[filteredCounter]
                filteredCounter += 1
            else:
                apDict[idx]['subhalo']['tau_v'] = np.nan

        print("Successfully assigned V-band optical depths to aperture dictionary!")

        return apDict

    def run_calibration(self, apDict, initialGuess=(0.2, 0.4), alpha=-0.6, maxiter=100):
        """
        Function:
            Executes the calibration of the V-band optical depths by executing the preceding methods
            Calculates optBeta and optGamma using compute_optimal_params
            Calculates V-band optical depths 
        
        
        """

        optBeta, optGamma = self.compute_optimal_params(initialGuess, alpha, maxiter)
        opt_tauV_ap       = self.compute_tauV_from_apDict(apDict, alpha, optBeta, optGamma)
        opt_tauV_cal      = self.compute_tauV_from_subDict(alpha, optBeta, optGamma)

        
        apDict = self.assign_optTauV(apDict, alpha, optBeta, optGamma)

        df_GSWLC_valid, GSWLC_tauV = self.compute_GSWLC_tauV()

        calibration_results = {
            'Optimal_Params': {'Beta': optBeta, 'Gamma': optGamma},
            'GSWLC': {'df': df_GSWLC_valid, 'tauV': GSWLC_tauV},
            'Aperture_Snapshot': {'Dict': apDict, 'tauV': opt_tauV_ap},
            'Calibration_Snapshot': {'Redshift': self.redshift, 'tauV': opt_tauV_cal}
        }

        return calibration_results
