import os

import pandas as pd
import numpy as np
import logging
from calibrate import CalibOutput
from scipy.optimize import fsolve, root, minimize, approx_fprime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import EquilibriumOutput, add_long_description, flatten_index, unflatten_index_in_df
from dataclasses import dataclass
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']
DIRTY_ENERGY_USE = ['Petro', 'FuelDist']

ENERGY_SECTORS = DIRTY_ENERGY_SECTORS +  ['Power']

DURABLE_GOODS = [ 'FabMetal', 'Machine', 'Vehicle', 'TransOth', 'Repair', 'Electronic', 'Electrical',
                  'FurnOth', 'ConstBuild', 'ConstCivil']
NON_DURABLE_GOODS = ['Wheat', 'Maize', 'CerOth', 'Legume', 'Rice', 'Veget', 'Sugar', 'Tobac', 'Fibre', 'CropOth',
                     'Grape', 'FruitNut', 'CropBev', 'SpicePhar', 'Seed', 'Cattle', 'Sheep', 'Pig', 'Poultry', 'AnimOth',
                     'Forest', 'Fish', 'Crust', 'IronOres', 'UranOres', 'AluOres', 'CopperOres', 'GoldOres',
                     'LedZinSilOres', 'NickelOres', 'TinOres', 'NonferOthOres', 'StoneSand', 'ChemFert', 'Salt',
                     'OthServ', 'BeefMeat', 'SheepMeat', 'PorkMeat', 'PoultryMeat', 'MeatOth', 'FishProd', 'Cereal',
                     'VegetProd', 'Fruit', 'FoodOth', 'Sweets', 'FatAnim', 'FatVeget', 'Dairy', 'Beverage',
                     'TobacProd', 'Textile', 'Leather', 'Sawmill', 'Paper', 'Print', 'NFert', 'OthFert', 'ChemPetro',
                     'ChemInorg', 'ChemOrg', 'Pharma', 'ChemOth', 'Rubber', 'Plastic', 'Clay', 'Ceramics', 'Cement', 'MinOth',
                     'IronSteel', 'Alu', 'Copper', 'Gold', 'LedZinSil', 'Nickel', 'Tin', 'NonferOth', 'Water', 'Waste',
                     'Recovery', 'TradeRep', 'Road', 'Rail', 'Pipe', 'WaterTrans', 'Air', 'Services', 'Post', 'Hospitality',
                     'Publishing', 'Telecom', 'Info', 'Finance', 'Real', 'ProfSci', 'Admin', 'Public', 'Edu', 'Health',
                     'Arts', 'Oth']


def read_file_shocks(path):
    """Reads the excel file defining the demand shocks. Those shocks include: sector-specific shocks, and nest-specific
    shocks. In particular, shocks are for the nest between energy and durable goods, and the nest between energy services
    and non-durable goods."""
    with pd.ExcelFile(path) as xls:
        shocks_sector = pd.read_excel(xls, sheet_name="sector", index_col=0, header=0)
        shocks_energy_durable = pd.read_excel(xls, sheet_name="energy_durable", index_col=0, header=0)
        shocks_nondurable_energyservices = pd.read_excel(xls, sheet_name="nondurable_energyservices", index_col=0, header=0)
        shocks_sector_IO = pd.read_excel(xls, sheet_name="sector_IO", index_col=0, header=0)
        shocks_energy_durable_IO = pd.read_excel(xls, sheet_name="energy_durable_IO", index_col=0, header=0)
        shocks_nondurable_energyservices_IO = pd.read_excel(xls, sheet_name="nondurable_energyservices_IO", index_col=0, header=0)
        shocks_sector_ROW = pd.read_excel(xls, sheet_name="sector_ROW", index_col=0, header=0)
        shocks_energy_durable_ROW = pd.read_excel(xls, sheet_name="energy_durable_ROW", index_col=0, header=0)
        shocks_nondurable_energyservices_ROW = pd.read_excel(xls, sheet_name="nondurable_energyservices_ROW", index_col=0, header=0)
    shocks = {
        'sector': shocks_sector,
        'energy_durable': shocks_energy_durable,
        'nondurable_energyservices': shocks_nondurable_energyservices,
        'sector_IO': shocks_sector_IO,
        'energy_durable_IO': shocks_energy_durable_IO,
        'nondurable_energyservices_IO': shocks_nondurable_energyservices_IO,
        'sector_ROW': shocks_sector_ROW,
        'energy_durable_ROW': shocks_energy_durable_ROW,
        'nondurable_energyservices_ROW': shocks_nondurable_energyservices_ROW,
    }
    return shocks


def process_shocks(col, shocks, uniform_shock, domestic_country, countries, new_consumer):
    """Selects the column corresponding to the type of the shock, and processes it to be used in the model.
    In particular, this involves specifying if the shock is specific to domestic country, or is shared across countries.
    uniform_shock: boolean, if True, the shock is shared across countries. If False, the shock is specific to the domestic country.
    domestic_country: str, the domestic country
    countries: list of str, the list of countries in the model.
    """
    list_key = ['sector', 'energy_durable', 'nondurable_energyservices', 'sector_IO', 'energy_durable_IO', 'nondurable_energyservices_IO']
    betai_hat_dict = {}
    for key in list_key:
        betai_hat = shocks[key]
        betai_hat = betai_hat[col]  # get specific shocks
        betai_hat = betai_hat.to_frame()
        if new_consumer:
            betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: f'{domestic_country}1'})
        else:
            betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: domestic_country})

        if uniform_shock:  # Preferences shocks are shared across countries
            if 'IO' in key:  # in this case, we can only use the same inputs for all countries
                betai_hat = pd.concat([betai_hat] * len(countries), axis=1)
            else:
                betai_hat_ROW = shocks[f'{key}_ROW']
                betai_hat_ROW = betai_hat_ROW[col]  # get specific shocks
                betai_hat_ROW = betai_hat_ROW.to_frame()
                if new_consumer:  # we have to differentiate between the new consumer from domestic country, and the RoW country
                    betai_hat = pd.concat([betai_hat]*(len(countries)-1) + [betai_hat_ROW], axis=1)
                else:
                    betai_hat = pd.concat([betai_hat, betai_hat_ROW], axis=1)
            betai_hat.columns = countries

        else:  # Preferences shocks are specific to domestic country
            betai_hat = betai_hat.reindex(countries, axis=1, fill_value=1.0)  # we assume that the new consumer does not experience any preference shock

        betai_hat.index.names = ['Sector']
        betai_hat.columns.names = ['Country']
        betai_hat_dict[key] = betai_hat

    # betai_hat = shocks[col]  # get specific shocks
    # betai_hat = betai_hat.to_frame()
    # if new_consumer:
    #     betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: f'{domestic_country}1'})
    # else:
    #     betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: domestic_country})
    #
    # if uniform_shock:
    #     # Preferences shocks are shared across countries
    #     betai_hat = pd.concat([betai_hat] * len(countries), axis=1)
    #     betai_hat.columns = countries
    #
    # else:
    #     # Preferences shocks are specific to domestic country
    #     betai_hat = betai_hat.reindex(countries, axis=1, fill_value=1.0)
    #     # if new_consumer:  # we want the second consumer to experience the same preference shocks as the first consumer
    #     #     betai_hat[f'{domestic_country}2'] = betai_hat[f'{domestic_country}1']
    #
    # betai_hat.index.names = ['Sector']
    # betai_hat.columns.names = ['Country']
    return betai_hat_dict


class OptimizationContext:
    """Class to solve the equilibrium of the model"""
    def __init__(self, li_hat,ki_hat, betai_hat, a_efficiency, theta,sigma,epsilon,delta, mu, nu, kappa, rho, sectors, xsi, psi, phi, costs_energy_final, psi_energy,
                 psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy,
                 Omega_non_energy, Domestic, Delta, share_GNE, domestic_country, singlefactor, new_consumer, share_new_consumer):
        self.li_hat = li_hat
        self.ki_hat = ki_hat
        self.betai_hat = betai_hat
        self.a_efficiency = a_efficiency
        self.theta = theta
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.rho = rho
        self.sectors = sectors
        self.xsi = xsi
        self.psi = psi
        self.phi = phi
        self.costs_energy_final = costs_energy_final
        self.psi_energy = psi_energy
        self.psi_non_energy = psi_non_energy
        self.costs_durable_final = costs_durable_final
        self.psi_durable = psi_durable
        self.psi_non_durable = psi_non_durable
        self.costs_energy_services_final = costs_energy_services_final
        self.Omega = Omega
        self.costs_energy = costs_energy
        self.Omega_energy = Omega_energy
        self.Omega_non_energy = Omega_non_energy
        self.Domestic = Domestic
        self.Delta = Delta
        self.share_GNE = share_GNE
        self.domestic_country = domestic_country
        self.singlefactor = singlefactor
        self.new_consumer = new_consumer
        self.share_new_consumer = share_new_consumer

    def residuals_wrapper(self, lvec):
        # Call the original function but only return the first output
        res, *_ = residuals(lvec, self.li_hat, self.ki_hat, self.betai_hat, self.a_efficiency, self.theta, self.sigma, self.epsilon, self.delta,
                            self.mu, self.nu, self.kappa, self.rho, self.sectors, self.xsi, self.psi, self.phi, self.psi_energy, self.psi_non_energy,
                            self.costs_durable_final, self.psi_durable, self.psi_non_durable, self.costs_energy_services_final, self.Omega, self.costs_energy,
                            self.Omega_energy, self.Omega_non_energy, self.Domestic, self.Delta,
                            self.share_GNE, singlefactor=self.singlefactor, domestic_country=self.domestic_country, new_consumer=self.new_consumer)
        return res

    def solve_equilibrium(self, initial_guess):
        """Solves the equilibrium, using exact hat algebra. Specify the method used to find the solution (which is
        equivalent to finding the zeros of the function."""
        t1 = time.time()
        # lvec_sol = root(self.residuals_wrapper, initial_guess, method=method)
        lvec_sol, method = self.find_root(initial_guess)
        t_m = time.time() - t1
        residual = (self.residuals_wrapper(lvec_sol.x) ** 2).sum()
        logging.info(f"Method: {method:10s}, Time: {t_m:5.1f}, Residual: {residual:10.2e}")
        res, output = residuals(lvec_sol.x, self.li_hat, self.ki_hat, self.betai_hat, self.a_efficiency, self.theta, self.sigma, self.epsilon,
                                self.delta, self.mu, self.nu, self.kappa, self.rho, self.sectors, self.xsi, self.psi, self.phi, self.psi_energy, self.psi_non_energy,
                                self.costs_durable_final, self.psi_durable, self.psi_non_durable, self.costs_energy_services_final, self.Omega, self.costs_energy,
                                self.Omega_energy, self.Omega_non_energy, self.Domestic, self.Delta,
                                self.share_GNE, singlefactor=self.singlefactor, domestic_country=self.domestic_country,
                                new_consumer=self.new_consumer)
        return lvec_sol.x, output

    def find_root(self, initial_guess):
        # return root(self.residuals_wrapper, initial_guess, method='krylov')
        for method in ['krylov', 'hybr', 'lm']:
            try:
                lvec_sol = root(self.residuals_wrapper, initial_guess, method=method)
                return lvec_sol, method
            except Exception as e:
                logging.info(f"Method {method} failed with an unexpected error: {e}. Trying next method...")
        raise ValueError("All methods failed.")


def residuals(lvec, li_hat, ki_hat, betai_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho, sectors, xsi, psi, phi, psi_energy,
              psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
              Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, singlefactor='specific', domestic_country = 'FRA',
              new_consumer=False):
    """Function to compute the residuals of the model. Residuals are obtained from FOC from the model. The goal is then to
    minimize this function in order to find its zeros, corresponding to the equilibrium.
    #######
    theta: elasticity between labor/capital and intermediate inputs for production
    sigma: elasticity between goods in final demand
    epsilon: elasticity between intermediate inputs in production
    delta: elasticity between goods in import nest for production
    mu: elasticity between goods in import nest for final demand
    nu: elasticity between energy and non-energy intermediate inputs in production
    kappa: elasticity between energy and non-energy goods in final demand
    """
    assert singlefactor in ['specific', 'country', 'all'], 'Parameter singlefactor is not correctly specified.'
    N = len(sectors)  # number of sectors times countries
    S = len(psi)  # number of sectors
    if new_consumer:  # we have one consumer more than the number of countries
        C = xsi.shape[1] - 1 # number of countries
        C_consumer = xsi.shape[1]
    else:
        C = xsi.shape[1]
        C_consumer = C
    vec = np.exp(lvec)
    pi_hat = vec[:N]
    yi_hat = vec[N:2*N]
    pi_hat = pd.Series(pi_hat, index=sectors.index)
    yi_hat = pd.Series(yi_hat, index=sectors.index)
    PSigmaY = pd.Series(index=psi.columns, data=vec[2*N:2*N+C_consumer])  #

    if singlefactor == 'country':  # factors can be freely reallocated inside each country
        w_country = pd.Series(index=psi.columns, data=vec[2*N+C_consumer:2*N+C_consumer+C])
        r_country = pd.Series(index=psi.columns, data=vec[2*N+C_consumer+C:2*N+C_consumer+2*C])
    elif singlefactor == 'all':  # factors can be freely reallocated across the world
        w_world = vec[2*N+C_consumer:2*N+C_consumer+1]
        r_world = vec[2*N+C_consumer+1:2*N+C_consumer+2]

    assert Domestic.index.equals(Domestic.columns), "Domestic index and columns are not the same while they should be"
    # We create a price dataframe with the same dimensions as the Domestic dataframe
    # With the following syntax, we assume that pi_hat follows the order of Domestic columns. Important !!

    # Price for imports nest
    if delta == 1:
        price_imports = Domestic * np.log(pi_hat)
        price_imports = np.exp(price_imports.groupby(level='Sector', axis=1).sum())
    else:
        price_imports = Domestic * pi_hat**(1-delta)
        price_imports = (price_imports.groupby(level='Sector', axis=1).sum())**(1/(1-delta))
    assert price_imports.shape == Omega.shape

    # Price for intermediate sectors goods, without distinction between energy and non-energy
    if epsilon == 1:
        price_intermediate = Omega * np.log(price_imports)
        price_intermediate = np.exp(price_intermediate.sum(axis=1))
    else:
        price_intermediate = Omega * price_imports**(1-epsilon)
        price_intermediate = price_intermediate.sum(axis=1)**(1/(1-epsilon))
    # price_intermediate = price_intermediate.to_frame()
    assert price_intermediate.shape == (N,)

    # Price for intermediate energy sectors goods
    if epsilon == 1:
        price_intermediate_energy = Omega_energy * np.log(price_imports)
        price_intermediate_energy = np.exp(price_intermediate_energy.sum(axis=1))
    else:
        price_intermediate_energy = Omega_energy * price_imports.loc[:,price_imports.columns.isin(ENERGY_SECTORS)]**(1-epsilon)
        price_intermediate_energy = price_intermediate_energy.sum(axis=1)**(1/(1-epsilon))

    # Price for intermediate non-energy sectors goods
    if epsilon == 1:
        price_intermediate_non_energy = Omega_non_energy * np.log(price_imports)
        price_intermediate_non_energy = np.exp(price_intermediate_non_energy.sum(axis=1))
    else:
        price_intermediate_non_energy = Omega_non_energy * price_imports.loc[:,~price_imports.columns.isin(ENERGY_SECTORS)]**(1-epsilon)
        price_intermediate_non_energy = price_intermediate_non_energy.sum(axis=1)**(1/(1-epsilon))

    # Intermediate price shared by all sectors, between energy and non-energy nest
    if nu == 1:
        # price_intermediate = (costs_energy.loc[:, costs_energy.columns.isin(ENERGY_SECTORS)].iloc[:,0] * np.log(price_intermediate_energy) + costs_energy.loc[:, ~costs_energy.columns.isin(ENERGY_SECTORS)].iloc[:,0] * np.log(price_intermediate_non_energy))
        price_intermediate = (costs_energy['Energy'] * np.log(price_intermediate_energy) + costs_energy['Non-Energy'] * np.log(price_intermediate_non_energy))
        price_intermediate = np.exp(price_intermediate)
    else:
        # price_intermediate = (costs_energy.loc[:, costs_energy.columns.isin(ENERGY_SECTORS)].iloc[:,0] * price_intermediate_energy**(1-nu) + costs_energy.loc[:, ~costs_energy.columns.isin(ENERGY_SECTORS)].iloc[:,0] * price_intermediate_non_energy**(1-nu))**(1/(1-nu))
        price_intermediate = (costs_energy['Energy'] * price_intermediate_energy**(1-nu) + costs_energy['Non-Energy'] * price_intermediate_non_energy**(1-nu))**(1/(1-nu))

    price_intermediate_energy_concat = pd.concat([price_intermediate_energy]*len(ENERGY_SECTORS), axis=1)  # this price is shared by all energy sectors
    price_intermediate_energy_concat.columns = ENERGY_SECTORS
    price_intermediate_non_energy_concat = pd.concat([price_intermediate_non_energy]*len(Omega_non_energy.columns ), axis=1)  # this price is shared by all non-energy sectors
    price_intermediate_non_energy_concat.columns = Omega_non_energy.columns
    price_intermediate_energy_overall = pd.concat([price_intermediate_energy_concat, price_intermediate_non_energy_concat], axis=1)  # we aggregate the price for energy and non-energy nest, for calculation purposes
    price_intermediate_energy_overall.columns.name = 'Sector'

    # Intermediate demand
    # intermediate_demand = (price_imports**(delta-epsilon)).mul(yi_hat * pi_hat**theta  * price_intermediate**(epsilon-theta), axis=0) * pi_hat**(-delta)  # old version without energy nest
    intermediate_demand = (price_intermediate_energy_overall**(epsilon - nu) * price_imports**(delta-epsilon)).mul(yi_hat * pi_hat**theta * price_intermediate**(nu-theta), axis=0) * pi_hat**(-delta)

    if singlefactor == 'country':  # li_hat and ki_hat are endogenous in this case
        vi_hat = np.exp(sectors['gamma'] * np.log(w_country) + (1 - sectors['gamma']) * np.log(r_country))
        hi_hat = yi_hat * (pi_hat / vi_hat) ** theta
        li_hat = hi_hat * vi_hat / w_country
        ki_hat = hi_hat * vi_hat / r_country
    elif singlefactor == 'all':
        vi_hat = np.exp(sectors['gamma'] * np.log(w_world) + (1 - sectors['gamma']) * np.log(r_world))
        hi_hat = yi_hat * (pi_hat / vi_hat) ** theta
        li_hat = hi_hat * vi_hat / w_world
        ki_hat = hi_hat * vi_hat / r_world
        pass
    else:  # sector specific factors
        hi_hat = li_hat**(sectors['gamma']) * ki_hat**(1-sectors['gamma'])
        vi_hat = pi_hat * (yi_hat / hi_hat)**(1/theta)
        wi_hat = vi_hat * hi_hat / li_hat
        ri_hat = vi_hat * hi_hat / ki_hat

    # Price for import nest, final demand
    if mu == 1:
        price_imports_finaldemand = xsi.mul(np.log(pi_hat), axis=0)
        price_imports_finaldemand = np.exp(price_imports_finaldemand.groupby(level='Sector').sum())
    else:
        price_imports_finaldemand = xsi.mul(pi_hat**(1-mu), axis=0)
        price_imports_finaldemand = (price_imports_finaldemand.groupby(level='Sector').sum())**(1/(1-mu))
    price_imports_finaldemand = price_imports_finaldemand.reindex(psi.index, axis=0)  # we reindex to ensure same order for interpretability
    assert price_imports_finaldemand.shape == psi.shape

    # Price for intermediate energy sectors goods in final demand
    if sigma == 1:
        pass
    else:
        price_index_energy = betai_hat['sector'].loc[betai_hat['sector'].index.isin(ENERGY_SECTORS), :] * psi_energy * (price_imports_finaldemand.loc[price_imports_finaldemand.index.isin(ENERGY_SECTORS), :])**(1-sigma)
        price_index_energy = price_index_energy.sum(axis=0)**(1/(1-sigma))

    # Price for intermediate durable sectors goods in final demand
    if sigma == 1:
        pass
    else:
        price_index_durable = betai_hat['sector'].loc[betai_hat['sector'].index.isin(DURABLE_GOODS), :] * psi_durable * (price_imports_finaldemand.loc[price_imports_finaldemand.index.isin(DURABLE_GOODS),:])**(1-sigma)
        price_index_durable = price_index_durable.sum(axis=0)**(1/(1-sigma))

    # Price for intermediate non durable sectors goods in final demand
    if sigma == 1:
        pass
    else:
        price_index_non_durable = betai_hat['sector'].loc[betai_hat['sector'].index.isin(NON_DURABLE_GOODS), :] * psi_non_durable * (price_imports_finaldemand.loc[price_imports_finaldemand.index.isin(NON_DURABLE_GOODS),:])**(1-sigma)
        price_index_non_durable = price_index_non_durable.sum(axis=0)**(1/(1-sigma))

    aeff = a_efficiency.loc[ENERGY_SECTORS].mean()
    if kappa == 1:
        pass
        # this setting is only specified when we assume NO PREFERENCES SHOCKS, BUT EFFICIENCY SHOCK
        # if aeff[domestic_country] != 1:
        #     price_index_energy_services = costs_energy_services_final.loc['Energy'] * np.log(price_index_energy) + costs_energy_services_final.loc['Durable'] * np.log(price_index_durable)
        #     price_index_energy_services = aeff ** (-costs_energy_services_final.loc['Energy']) * np.exp(price_index_energy_services)
        # else:
        #     print(f'Error, model is not specified for exact Cobb Douglas setting when efficiency improvement is zero.')
    else:
        price_index_energy_services = (betai_hat['energy_durable'].loc['Energy'] * costs_energy_services_final.loc['Energy'] * (price_index_energy / aeff) ** (1 - kappa) + betai_hat['energy_durable'].loc['Durable'] * costs_energy_services_final.loc['Durable'] * price_index_durable ** (1 - kappa)) ** (1 / (1 - kappa))

    if rho == 1:
        pass
    else:
        price_index = (betai_hat['nondurable_energyservices'].loc['Non-Durable'] * costs_durable_final.loc['Non-Durable'] * price_index_non_durable ** (1-rho) + betai_hat['nondurable_energyservices'].loc['Energy-Services'] * costs_durable_final.loc['Energy-Services'] * price_index_energy_services ** (1-rho)) ** (1 / (1-rho))

    price_index_intermediary_nests_concat = pd.DataFrame({sector: price_index_energy_services ** (kappa - rho) * price_index_energy ** (sigma - kappa) if sector in ENERGY_SECTORS else (price_index_energy_services ** (kappa - rho) * price_index_durable ** (sigma - kappa) if sector in DURABLE_GOODS else price_index_non_durable ** (sigma - rho)) for sector in xsi.index.get_level_values('Sector').unique()}).T
    price_index_intermediary_nests_concat.index.name = 'Sector'

    betai_energydurable_hat_concat = pd.DataFrame({sector: betai_hat['energy_durable'].loc['Energy'] if sector in ENERGY_SECTORS else (betai_hat['energy_durable'].loc['Durable'] if sector in DURABLE_GOODS else 1) for sector in xsi.index.get_level_values('Sector').unique()}).T
    betai_energydurable_hat_concat.index.name = 'Sector'

    betai_nondurableenergyservices_hat_concat = pd.DataFrame({sector: betai_hat['nondurable_energyservices'].loc['Non-Durable'] if sector in NON_DURABLE_GOODS else betai_hat['nondurable_energyservices'].loc['Energy-Services'] for sector in xsi.index.get_level_values('Sector').unique()}).T
    betai_nondurableenergyservices_hat_concat.index.name = 'Sector'

    # Final demand
    # final_demand = (a_efficiency**(kappa-1) * betai_hat * PSigmaY * price_index**(kappa) *  price_index_energy_concat**(sigma - kappa) * price_imports_finaldemand**(mu - sigma) ).mul(pi_hat**(-mu), axis=0)
    # final_demand = (a_efficiency ** (kappa-1) * betai_hat['sector'] * betai_energydurable_hat_concat * betai_nondurableenergyservices_hat_concat * PSigmaY * price_index**rho * price_index_energy_services_concat ** (kappa - rho) * price_index_energy_concat ** (sigma - kappa) * price_imports_finaldemand ** (mu - sigma)).mul(pi_hat ** (-mu), axis=0)
    final_demand = (a_efficiency ** (kappa-1) * betai_hat['sector'] * betai_energydurable_hat_concat * betai_nondurableenergyservices_hat_concat * PSigmaY * price_index ** rho * price_index_intermediary_nests_concat * price_imports_finaldemand ** (mu - sigma)).mul(pi_hat ** (-mu), axis=0)


    ### Residuals

    nominal_GDP = PSigmaY * price_index

    # Prices
    if theta == 1:
        res1 = np.log(pi_hat) - (sectors['eta'] * np.log(vi_hat) + (1 - sectors['eta']) * np.log(price_intermediate))
    else:
        res1 = pi_hat ** (1 - theta) - (sectors['eta'] * vi_hat ** (1 - theta) + (1 - sectors['eta']) * price_intermediate ** (1 - theta))

    # Quantities
    res2 = yi_hat - ((final_demand * phi).sum(axis=1) + (Delta * intermediate_demand).sum(axis=0))

    if new_consumer:  # we define the revenues from the new consumer
        res3bis = nominal_GDP[f'{domestic_country}2'] - nominal_GDP[f'{domestic_country}1']
        nominal_GDP[domestic_country] = nominal_GDP[f'{domestic_country}2']
        nominal_GDP = nominal_GDP.drop([f'{domestic_country}2', f'{domestic_country}1'])

    # World GDP is the numeraire, and stays the same
    res3 = (share_GNE * nominal_GDP).sum() - 1


    revenue = sectors.loc[:, sectors.columns.str.contains('rev_')]
    rev_labor_dom = revenue['rev_labor'].xs(domestic_country, level='Country', axis=0)
    rev_capital_dom = revenue['rev_capital'].xs(domestic_country, level='Country', axis=0)
    # TODO: this line should be modified if I include more countries than just two
    li_hat_dom = li_hat.xs(domestic_country, level='Country', axis=0)
    ki_hat_dom = ki_hat.xs(domestic_country, level='Country', axis=0)

    if singlefactor == 'country':  # factors can be freely reallocated inside each country
        wi_hat_dom = w_country[domestic_country]
        ri_hat_dom = r_country[domestic_country]
        res4 = (rev_labor_dom  * li_hat_dom  * wi_hat_dom).sum() + (rev_capital_dom  * ki_hat_dom  * ri_hat_dom).sum() - (nominal_GDP)[domestic_country]

        share_labor = sectors['share_labor']
        share_capital = sectors['share_capital']
        # labor_shock = pd.Series(index=['EUR', 'ROW'], inputs=[1,1])  # labor supply shock
        res5 = 1 - (share_labor * li_hat).groupby(level="Country").sum()
        res6 = 1 - (share_capital * ki_hat).groupby(level="Country").sum()

        res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res5.to_numpy(), res6.to_numpy()])

    elif singlefactor == 'all':  # factors can be freely reallocated across the world
        res4 = (rev_labor_dom * li_hat_dom * w_world).sum() + (rev_capital_dom * ki_hat_dom * r_world).sum() - (nominal_GDP)[domestic_country]
        share_labor_tot = sectors['share_labor_tot']
        share_capital_tot = sectors['share_capital_tot']
        res5 = 1 - (share_labor_tot * li_hat).sum()
        res6 = 1 - (share_capital_tot * ki_hat).sum()

        res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), np.array([res5]), np.array([res6])])

    else:  # sector specific factors
        # Definition of GDP in other country
        wi_hat_dom = wi_hat.xs(domestic_country, level='Country', axis=0)
        ri_hat_dom = ri_hat.xs(domestic_country, level='Country', axis=0)
        res4 =(rev_labor_dom  * li_hat_dom  * wi_hat_dom).sum() + (rev_capital_dom  * ki_hat_dom  * ri_hat_dom).sum() - (nominal_GDP)[domestic_country]

        res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4])])

    if new_consumer:
        res = np.concatenate([res, np.array([res3bis])])

    #
    # TODO: this aggregation is no longer valid, due to the new energy nest. Should be done again.
    final_demand_aggregator = ((xsi * final_demand**((mu-1)/mu)).groupby('Sector').sum())**(mu/(mu-1))
    budget_shares_hat = final_demand_aggregator * price_imports_finaldemand / (PSigmaY*price_index)
    budget_shares_new = budget_shares_hat * psi  # new budget shares, compiled from the hat and the initial version
    variation_welfare = np.log(nominal_GDP) + np.log(((budget_shares_new*price_imports_finaldemand**(sigma - 1)).sum())**(1/(1-sigma)))

    # New budget shares with my new concatenation for energy services, durable and non-durable goods
    costs_durable_final_concat = pd.DataFrame({sector: costs_durable_final.loc['Non-Durable' if sector in NON_DURABLE_GOODS else 'Energy-Services'] for sector in xsi.index.get_level_values('Sector').unique()}).T
    costs_durable_final_concat.index.name = 'Sector'
    costs_energy_services_final_concat = pd.DataFrame({sector: costs_energy_services_final.loc['Energy'] if sector in ENERGY_SECTORS else  (costs_energy_services_final.loc['Durable'] if sector in DURABLE_GOODS else 1) for sector in xsi.index.get_level_values('Sector').unique()}).T
    costs_energy_services_final_concat.index.name = 'Sector'
    budget_shares = xsi * pd.concat([psi_durable, psi_non_durable, psi_energy], axis=0) * costs_durable_final_concat * costs_energy_services_final_concat  # initial budget shares for each of the sectors, where we create a new budget share for energy shared across sectors
    expenditure_share_variation = final_demand.mul(pi_hat, axis=0) / (PSigmaY*price_index)
    tornqvist_price_index = np.exp((budget_shares * (1+expenditure_share_variation) / 2).mul(np.log(pi_hat), axis=0).sum(axis=0))
    sato_vartia_price_index = (-budget_shares * (1 - expenditure_share_variation) / np.log(expenditure_share_variation)).where(expenditure_share_variation != 1, other=budget_shares)  # when expenditure did not change, the value is the initial budget share
    sato_vartia_price_index = sato_vartia_price_index / sato_vartia_price_index.sum(axis=0)  # we use the formula from the foundational paper from Sato and Vartia (1976)
    sato_vartia_price_index = np.exp(sato_vartia_price_index.mul(np.log(pi_hat), axis=0).sum(axis=0))  # we again calculate the log price index

    # TODO: Need to integrate efficiency parameter here, only works when aeff = 1 for now. Il faudrait l'ajouter dans l'aggrégateur pour les energy services
    if kappa != 1:
        final_demand_aggregator = ((xsi * final_demand ** ((mu - 1) / mu)).groupby('Sector').sum()) ** (mu / (mu - 1))
        final_demand_aggregator = (pd.concat([psi_durable, psi_non_durable, psi_energy], axis=0) * betai_hat['sector']**(1/sigma) * final_demand_aggregator ** ((sigma-1) / sigma)).groupby(lambda x: 'Energy' if x in ENERGY_SECTORS else ('Durable' if x in DURABLE_GOODS else 'Non-Durable')).sum() ** (sigma / (sigma - 1))
        final_demand_aggregator = pd.concat([final_demand_aggregator.loc['Non-Durable',:], ((betai_hat['energy_durable']**(1/kappa) * costs_energy_services_final * final_demand_aggregator.loc[['Energy', 'Durable'],:] ** ((kappa-1) / kappa)).sum(axis=0)) ** (kappa / (kappa-1))], axis=1).rename(columns={0: 'Energy-Services'}).T
        budget_shares_new = costs_durable_final * final_demand_aggregator * pd.concat([price_index_energy_services.rename('Energy-Services'), price_index_non_durable.rename('Non-Durable')], axis=1).T / (PSigmaY*price_index)
        lloyd_moulton_price_index = (((budget_shares_new * (pd.concat([price_index_energy_services.rename('Energy-Services'), price_index_non_durable.rename('Non-Durable')], axis=1).T) ** (rho - 1)).sum(axis=0)) ** (1 / (rho - 1)))
    else:  # TODO: a modifier, pour debug
        lloyd_moulton_price_index = pd.Series(1, index=xsi.columns)


    #
    output = {
        'pi_hat': pi_hat,
        'yi_hat': yi_hat,
        'price_intermediate_energy': price_intermediate_energy,
        'price_intermediate_non_energy': price_intermediate_non_energy,
        'price_intermediate': price_intermediate,
        'li_hat': li_hat,
        'ki_hat': ki_hat,
        'PSigmaY': PSigmaY,
        'pi_imports_finaldemand': price_imports_finaldemand,
        'price_index': price_index,
        'price_index_energy': price_index_energy,
        'price_index_durable': price_index_durable,
        'price_index_non_durable': price_index_non_durable,
        'price_index_energy_services': price_index_energy_services,
        'tornqvist_price_index': tornqvist_price_index,
        'sato_vartia_price_index': sato_vartia_price_index,
        'lloyd_moulton_price_index': lloyd_moulton_price_index,
        'GDP': nominal_GDP,
        'domestic_domar': pi_hat * yi_hat / (nominal_GDP),
        'domar': pi_hat * yi_hat,
        'final_demand': final_demand,
        'intermediate_demand': intermediate_demand,
        'variation_welfare': variation_welfare
    }
    if singlefactor == 'country':
        output['w_country'] = w_country
        output['r_country'] = r_country
        output['domestic_factor_labor_domar'] = li_hat * w_country / output['GDP']
        output['domestic_factor_capital_domar'] = ki_hat * r_country / output['GDP']
        output['factor_labor_domar'] = li_hat * w_country
        output['factor_capital_domar'] = ki_hat * r_country
    elif singlefactor == 'all':
        output['w_world'] = w_world
        output['r_world'] = r_world
        output['domestic_factor_labor_domar'] = li_hat * w_world / output['GDP']
        output['domestic_factor_capital_domar'] = ki_hat * r_world / output['GDP']
        output['factor_labor_domar'] = li_hat * w_world
        output['factor_capital_domar'] = ki_hat * r_world
    else:
        output['wi_hat'] = wi_hat
        output['ri_hat'] = ri_hat
        output['domestic_factor_labor_domar'] = li_hat * wi_hat / output['GDP']
        output['domestic_factor_capital_domar'] = ki_hat * ri_hat / output['GDP']
        output['factor_labor_domar'] = li_hat * wi_hat
        output['factor_capital_domar'] = ki_hat * ri_hat
    return res, output


def run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy,
                    costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy,
                    Gamma, Leontieff, Domestic, Delta, sectors_dirty_energy,
                    final_use_dirty_energy, share_GNE, domestic_country, descriptions, reference_parameters,
                    new_consumer, share_new_consumer):
    """Solves the equilibrium, under different settings."""

    if new_consumer:
        C = xsi.shape[1] - 1
        C_consumer = C + 1
    else:
        C = xsi.shape[1]
        C_consumer = C

    theta, sigma, epsilon, delta, mu, nu, kappa, rho = reference_parameters['theta'], reference_parameters['sigma'], reference_parameters['epsilon'], reference_parameters['delta'], reference_parameters['mu'], reference_parameters['nu'], reference_parameters['kappa'], reference_parameters['rho']

    N = len(sectors)

    singlefactor = 'specific'

    logging.info('Solving for reference')
    context_ref = OptimizationContext(li_hat, ki_hat, betai_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho,
                                      sectors, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final,
                                      psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                                      Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country, singlefactor, new_consumer, share_new_consumer)
    initial_guess = np.zeros(2 * N + C_consumer)
    sol, output_ref = context_ref.solve_equilibrium(initial_guess)

    logging.info('Solving for Cobb-Douglas')
    elasticity_cb = 0.97
    context = OptimizationContext(li_hat, ki_hat, betai_hat,
                                  a_efficiency, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb,elasticity_cb,
                                  sectors, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final,
                                  psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                                  Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country, singlefactor, new_consumer, share_new_consumer)
    initial_guess = np.zeros(2 * N + C_consumer)
    sol_CD, output_CD = context.solve_equilibrium(initial_guess)

    logging.info('Solving for single factor shared across the world')
    singlefactor = 'all'
    context_single = OptimizationContext(li_hat, ki_hat, betai_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho,
                                         sectors, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final,
                                         psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                                         Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country, singlefactor, new_consumer, share_new_consumer)
    initial_guess = np.zeros(2 * N + C_consumer + C)
    sol, output_single = context_single.solve_equilibrium(initial_guess)

    # logging.info('Solving for single factor')
    # singlefactor = 'country'
    # context_single = OptimizationContext(li_hat, ki_hat, betai_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho,
    #                                      sectors, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final,
    #                                      psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
    #                                      Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country, singlefactor, new_consumer, share_new_consumer)
    # initial_guess = np.zeros(2 * N + C_consumer + 2*C)
    # sol, output_single = context_single.solve_equilibrium(initial_guess)



    output_dict = {
        'CD': output_CD,
        'ref': output_ref,
        'single': output_single
    }

    pi_hat = pd.concat([output_CD['pi_hat'].to_frame(name='price_CD_hat'), output_ref['pi_hat'].to_frame(name='price_hat'), output_single['pi_hat'].to_frame(name='price_single_hat')], axis=1)
    yi_hat = pd.concat([output_CD['yi_hat'].to_frame(name='quantity_CD_hat'), output_ref['yi_hat'].to_frame(name='quantity_hat'), output_single['yi_hat'].to_frame(name='quantity_single_hat')], axis=1)
    final_demand = pd.concat([output_CD['final_demand'].rename(columns={col: f'{col}_CD' for col in output_CD['final_demand'].columns}), output_ref['final_demand'].rename(columns={col: f'{col}' for col in output_ref['final_demand'].columns}),
         output_single['final_demand'].rename(columns={col: f'{col}_single' for col in output_single['final_demand'].columns})], axis=1)
    pi_imports_finaldemand = pd.concat([output_CD['pi_imports_finaldemand'].rename(columns={col: f'{col}_CD' for col in output_CD['pi_imports_finaldemand'].columns}), output_ref['pi_imports_finaldemand'].rename(columns={col: f'{col}' for col in output_ref['pi_imports_finaldemand'].columns}),
         output_single['pi_imports_finaldemand'].rename(columns={col: f'{col}_single' for col in output_single['pi_imports_finaldemand'].columns})], axis=1)
    domestic_domar = pd.concat([output_CD['domestic_domar'].to_frame(name='domestic_domar_CD_hat'), output_ref['domestic_domar'].to_frame(name='domestic_domar_hat'), output_single['domestic_domar'].to_frame(name='domestic_domar_single_hat')], axis=1)
    domar = pd.concat([output_CD['domar'].to_frame(name='domar_CD_hat'), output_ref['domar'].to_frame(name='domar_hat'), output_single['domar'].to_frame(name='domar_single_hat')], axis=1)
    domestic_factor_labor_domar = pd.concat([output_CD['domestic_factor_labor_domar'].to_frame(name='domestic_factor_labor_domar_CD_hat'), output_ref['domestic_factor_labor_domar'].to_frame(name='domestic_factor_labor_domar_hat'), output_single['domestic_factor_labor_domar'].to_frame(name='domestic_factor_labor_domar_single_hat')], axis=1)
    factor_labor_domar = pd.concat([output_CD['factor_labor_domar'].to_frame(name='factor_labor_domar_CD_hat'), output_ref['factor_labor_domar'].to_frame(name='factor_labor_domar_hat'), output_single['factor_labor_domar'].to_frame(name='factor_labor_domar_single_hat')], axis=1)
    domestic_factor_capital_domar = pd.concat([output_CD['domestic_factor_capital_domar'].to_frame(name='domestic_factor_capital_domar_CD_hat'), output_ref['domestic_factor_capital_domar'].to_frame(name='domestic_factor_capital_domar_hat'), output_single['domestic_factor_capital_domar'].to_frame(name='domestic_factor_capital_domar_single_hat')], axis=1)
    factor_capital_domar = pd.concat([output_CD['factor_capital_domar'].to_frame(name='factor_capital_domar_CD_hat'), output_ref['factor_capital_domar'].to_frame(name='factor_capital_domar_hat'), output_single['factor_capital_domar'].to_frame(name='factor_capital_domar_single_hat')], axis=1)
    domar_tot = pd.concat([domestic_domar, domar, domestic_factor_labor_domar, factor_labor_domar, domestic_factor_capital_domar, factor_capital_domar], axis=1)
    real_GDP = pd.concat([output_CD['PSigmaY'].rename({i: f'real_GDP_{i}_CD' for i in output_CD['PSigmaY'].index}), output_ref['PSigmaY'].rename({i: f'real_GDP_{i}_ref' for i in output_ref['PSigmaY'].index}), output_single['PSigmaY'].rename({i: f'real_GDP_{i}_single' for i in output_single['PSigmaY'].index})])
    GDP = pd.concat([output_CD['GDP'].rename({i: f'GDP_{i}_CD' for i in output_CD['GDP'].index}), output_ref['GDP'].rename({i: f'GDP_{i}_ref' for i in output_ref['GDP'].index}), output_single['GDP'].rename({i: f'GDP_{i}_single' for i in output_single['GDP'].index})])
    price_list_finaldemand = ['price_index', 'price_index_energy', 'price_index_durable', 'price_index_non_durable', 'price_index_energy_services',
                  'tornqvist_price_index', 'sato_vartia_price_index', 'lloyd_moulton_price_index']
    concatenated_price_info_finaldemand = {
        item: pd.concat(
            [df[item].rename(lambda i: f"{item}_{i}_{suffix}")
             for suffix, df in output_dict.items()]
        ) for item in price_list_finaldemand
    }
    price_index_finaldemand = pd.concat([concatenated_price_info_finaldemand[key] for key in concatenated_price_info_finaldemand.keys()], axis=0)
    price_list_production = ['price_intermediate_energy', 'price_intermediate_non_energy', 'price_intermediate']
    concatenated_price_info_production = {
        item: pd.concat(
            [df[item].to_frame().rename({0: f'{item}_{suffix}'}, axis=1)
             for suffix, df in output_dict.items()], axis=1
        ) for item in price_list_production
    }
    price_production = pd.concat([concatenated_price_info_production[key]for key in concatenated_price_info_production.keys()], axis=1)
    labor_capital_info = {
        item: pd.concat(
            [df[item].rename(f"{item}_{suffix}")
             for suffix, df in output_dict.items()], axis=1
        ) for item in ['li_hat', 'ki_hat']
    }
    labor_capital = pd.concat([labor_capital_info[key] for key in labor_capital_info.keys()], axis=1)
    variation_welfare = pd.concat([output_CD['variation_welfare'].rename({i: f'variation_welfare_{i}_CD' for i in output_CD['variation_welfare'].index}), output_ref['variation_welfare'].rename({i: f'variation_welfare_{i}_ref' for i in output_ref['variation_welfare'].index}), output_single['variation_welfare'].rename({i: f'variation_welfare_{i}_single' for i in output_single['variation_welfare'].index})])
    global_variables = pd.concat([GDP, real_GDP, price_index_finaldemand, variation_welfare], axis=0)

    emissions_hat, emissions_detail = variation_emission(output_dict, betai_hat, emissions, Leontieff, Gamma, phi, xsi, sectors, sectors_dirty_energy, final_use_dirty_energy, new_consumer, domestic_country, share_new_consumer)  # TODO: à modifier avec les betaihat

    equilibrium_output = EquilibriumOutput(pi_hat, yi_hat, price_production, pi_imports_finaldemand, final_demand, output_ref['intermediate_demand'], domar_tot, labor_capital, emissions_hat, emissions_detail, global_variables, descriptions)
    return equilibrium_output


def get_emissions_hat(yi_hat, intermediate_demand, final_demand, sectors_dirty_energy, final_use_dirty_energy, emissions,
                      new_consumer, country, share_new_consumer):
    """Computes variation in emissions based on variation of production, intermediate consumption and final demand.
    We rely on some approximations for this calculation. In particular, we do not distinguish between different types of dirty
    energy, and we only estimate an average variation in intermediate demand and final demand for these dirty energies. This
    is because it is hard to estimate those from the initial EDGAR-related database."""
    intermediate_demand_energy_dirty = intermediate_demand.loc[:,
                                       intermediate_demand.columns.get_level_values(1).isin(DIRTY_ENERGY_SECTORS)]
    intermediate_demand_energy_dirty = (intermediate_demand_energy_dirty * sectors_dirty_energy).sum(
        axis=1)  # average variation of intermediate consumption of dirty energy for each sector
    variation_emissions_energy = intermediate_demand_energy_dirty * emissions[
        'share_energy_related']  # variation of emissions related to energy
    variation_emissions_process = yi_hat * (
                1 - emissions['share_energy_related'])  # variation of emissions related to processes
    final_demand_energy = final_demand.loc[final_demand.index.get_level_values('Sector').isin(DIRTY_ENERGY_SECTORS), :]
    total_variation_emissions = (emissions['share_emissions_total_sectors'] * (
                variation_emissions_energy + variation_emissions_process)).groupby('Country').sum()
    final_demand_energy = (final_demand_energy * final_use_dirty_energy).sum(
        axis=0)  # average variation of final demand of dirty energy
    # For final demand, we only consider the share of total emissions, not differentiated by fossil fuel
    share_emissions_final_demand = emissions['share_emissions_total_finaldemand'].unstack('Country').sum(axis=0)
    if new_consumer:  # we modify the share of emissions between the two types of consumers
        share_emissions_final_demand['new_consumer'] = share_emissions_final_demand[country]
        share_emissions_final_demand = share_emissions_final_demand.rename({country: f'{country}1', 'new_consumer': f'{country}2'})
        share_emissions_final_demand[f'{country}2'] = share_new_consumer * share_emissions_final_demand[f'{country}2']
        share_emissions_final_demand[f'{country}1'] = (1 - share_new_consumer) * share_emissions_final_demand[f'{country}1']
    if new_consumer:
        group_mapping = {index: (country if index.startswith(country) else index) for index in share_emissions_final_demand.index}
        total_variation_emissions += (share_emissions_final_demand * final_demand_energy).groupby(group_mapping).sum()
    else:
        total_variation_emissions += share_emissions_final_demand * final_demand_energy
    emissions_breakdown = get_emissions_breakdown(yi_hat, intermediate_demand_energy_dirty, final_demand_energy, emissions, share_emissions_final_demand, country, new_consumer)
    return total_variation_emissions, emissions_breakdown


def get_emissions_breakdown(yi_hat, intermediate_demand_energy_dirty, final_demand_energy, emissions, share_emissions_final_demand, country, new_consumer):
    """Extracts the breakdown of emissions variation, differentiating between process-related, energy-related and final demand-related emissions. Also includes the specific sector
    from which the emissions variation originates."""
    variation_emissions_process = (emissions['share_emissions_total_sectors'] * (yi_hat - 1) * (1 - emissions['share_energy_related'])).unstack('Country')
    variation_emissions_process.index = pd.MultiIndex.from_tuples([(i, 'process') for i in variation_emissions_process.index], names=['Sector', 'Type'])
    variation_emissions_energy = (emissions['share_emissions_total_sectors'] * (intermediate_demand_energy_dirty - 1) * emissions['share_energy_related']).unstack('Country')
    variation_emissions_energy.index = pd.MultiIndex.from_tuples([(i, 'energy') for i in variation_emissions_energy.index], names=['Sector', 'Type'])
    if new_consumer:  
        group_mapping = {index: (country if index.startswith(country) else index) for index in share_emissions_final_demand.index}
        variation_emissions_final_demand = ((share_emissions_final_demand * (final_demand_energy - 1)).groupby(group_mapping).sum()).to_frame().T
        variation_emissions_final_demand.index = pd.MultiIndex.from_tuples([('FinalDemand', 'final demand')], names=['Sector', 'Type'])
    else:
        variation_emissions_final_demand = (share_emissions_final_demand * (final_demand_energy -1)).to_frame().T
        variation_emissions_final_demand.index = pd.MultiIndex.from_tuples([('FinalDemand', 'final demand')], names=['Sector', 'Type'])

    emissions_breakdown = pd.concat([variation_emissions_process, variation_emissions_energy, variation_emissions_final_demand], axis=0)
    return emissions_breakdown



def get_emissions_total(total_variation_emissions, emissions):
    """Computes absolute variations of emissions, based on relative variations, and inputs on world emissions."""
    tmp = total_variation_emissions.copy()  # relative variation of domestic emissions
    world_emissions = emissions[['total_sectors', 'final_demand']].sum().sum()
    total_variation_emissions.loc['total'] = (total_variation_emissions.squeeze() * (
                emissions[['total_sectors', 'final_demand']].groupby('Country').sum().sum(
                    axis=1) / world_emissions)).sum()  # we add total variation accounting for domestic and ROW emissions
    absolute_emissions = (tmp - 1).squeeze() * (
        emissions[['total_sectors', 'final_demand']].groupby('Country').sum().sum(
            axis=1))  # absolute variation of GHG emissions
    absolute_emissions.index = [f'{i}_absolute' for i in absolute_emissions.index]
    return absolute_emissions

def variation_emission(output_dict, betai_hat, emissions, Leontieff, Gamma, phi, xsi, sectors, sectors_dirty_energy, final_use_dirty_energy, new_consumer, domestic_country, share_new_consumer):
    """Estimations variation in emissions in the new equilibrium, compared to reference. Returns both aggregate emissions, and
    a breakdown of emissions decrease through sectors and processes."""
    results, emissions_detail_df = pd.DataFrame(), []
    betai_energydurable_hat_concat = pd.DataFrame({sector: betai_hat['energy_durable_IO'].loc['Energy'] if sector in ENERGY_SECTORS else (betai_hat['energy_durable_IO'].loc['Durable'] if sector in DURABLE_GOODS else 1) for sector in xsi.index.get_level_values('Sector').unique()}).T
    betai_energydurable_hat_concat.index.name = 'Sector'

    betai_nondurableenergyservices_hat_concat = pd.DataFrame({sector: betai_hat['nondurable_energyservices_IO'].loc['Non-Durable'] if sector in NON_DURABLE_GOODS else betai_hat['nondurable_energyservices_IO'].loc['Energy-Services'] for sector in xsi.index.get_level_values('Sector').unique()}).T
    betai_nondurableenergyservices_hat_concat.index.name = 'Sector'

    # For the input-output channel, we only consider some of the preference shocks, to avoid counting income rebound effect
    total_final_demand_variation = betai_hat['sector_IO'] * betai_energydurable_hat_concat * betai_nondurableenergyservices_hat_concat  # we consider variations across all nests to characterize total final variation
    total_variation_emissions_io, emissions_breakdown_io = input_output_calculation(total_final_demand_variation, Leontieff, Gamma, phi, sectors, sectors_dirty_energy, final_use_dirty_energy, emissions, new_consumer, domestic_country, share_new_consumer)
    total_variation_emissions_io = total_variation_emissions_io.to_frame().rename(columns={0: f'emissions_IO'})
    emissions_breakdown_io.columns = [f'{idx}_IO' for idx in emissions_breakdown_io.columns]
    emissions_detail_df.append(emissions_breakdown_io)
    absolute_emissions_io = get_emissions_total(total_variation_emissions_io, emissions)
    total_variation_emissions_io = pd.concat([total_variation_emissions_io, absolute_emissions_io.to_frame().rename(columns={0: f'emissions_IO'})], axis=0)
    results = pd.concat([results, total_variation_emissions_io], axis=1)
    for key, output in output_dict.items():
        intermediate_demand = output['intermediate_demand']
        yi_hat = output['yi_hat']
        final_demand = output['final_demand']
        total_variation_emissions, emissions_breakdown = get_emissions_hat(yi_hat, intermediate_demand, final_demand, sectors_dirty_energy, final_use_dirty_energy, emissions, new_consumer, domestic_country, share_new_consumer)
        total_variation_emissions = total_variation_emissions.to_frame().rename(columns={0: f'emissions_{key}'})
        emissions_breakdown.columns = [f'{idx}_{key}' for idx in emissions_breakdown.columns]
        emissions_detail_df.append(emissions_breakdown)
        absolute_emissions = get_emissions_total(total_variation_emissions, emissions)
        total_variation_emissions = pd.concat([total_variation_emissions, absolute_emissions.to_frame().rename(columns={0: f'emissions_{key}'})], axis=0)

        results = pd.concat([results, total_variation_emissions], axis=1)
    emissions_detail_df = pd.concat(emissions_detail_df, axis=1)
    return results, emissions_detail_df


def input_output_calculation(betai_hat, Leontieff, Gamma, phi, sectors, sectors_dirty_energy, final_use_dirty_energy, emissions,
                             new_consumer, domestic_country, share_new_consumer):
    """Takes vector of shocks and outputs estimated variation in emissions from simple input output framework.
    This framework does not include any price effect, and only relies on Leontieff matrix and direct input coefficients
    matrix."""
    # We calculate share of production that stems from final demand
    # phi = sectors.loc[:, sectors.columns.str.contains('phi_')]  # share of final consumption in total output
    # phi = phi.rename(columns=lambda x: x.split('_')[1])
    # phi.columns.names = ['Country']
    pyi = sectors.loc[:, 'pyi']
    final_use_init = phi.mul(pyi, axis=0)  # final use expenditures in the initial state

    final_use_new = betai_hat.copy() * final_use_init  # we calculate new final use

    pyi_new = Leontieff.mul(final_use_new.sum(axis=1), axis=0).sum(axis=0)  # we calculate required production from the Leontieff accounting equation y = L^T b
    yi_hat = pyi_new / pyi  # we calculate the relative variation for y, assuming there are no price changes in this setting

    final_demand = betai_hat.copy()
    intermediate_demand_hat = pd.concat([yi_hat]*len(Leontieff), axis=1)  # in leontieff model, intermediate inputs are used in fixed proportions
    intermediate_demand_hat.columns = Gamma.columns
    variation_emissions, emissions_breakdown = get_emissions_hat(yi_hat, intermediate_demand_hat, final_demand, sectors_dirty_energy, final_use_dirty_energy, emissions,
                                            new_consumer, country=domestic_country, share_new_consumer=share_new_consumer)
    return variation_emissions, emissions_breakdown


if __name__ == '__main__':

    code_country = {
        'france': 'FRA',
        'united_states_of_america': 'USA',
        'europe': 'EUR'
    }
    country = 'europe'
    domestic_country = code_country[country]
    filename = f"outputs/calib_{country}.xlsx"
    fileshocks = "inputs/shocks_demand_09042024.xlsx"
    uniform_shock = False
    new_consumer = False
    share_new_consumer = 0.5

    calib = CalibOutput.from_excel(filename)
    if new_consumer:
        calib.add_final_consumer(country=domestic_country, share_new_consumer=share_new_consumer)
    sectors, emissions, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, descriptions = calib.sectors, calib.emissions, calib.xsi, calib.psi, calib.phi, calib.costs_energy_final, calib.psi_energy, calib.psi_non_energy, calib.costs_durable_final, calib.psi_durable, calib.psi_non_durable, calib.costs_energy_services_final, calib.Omega, calib.costs_energy, calib.Omega_energy, calib.Omega_non_energy ,calib.Gamma, calib.Leontieff, calib.Domestic, calib.Delta, calib.sectors_dirty_energy, calib.final_use_dirty_energy, calib.share_GNE, calib.descriptions
    N = len(sectors)

    shocks = read_file_shocks(fileshocks)

    for col in shocks['sector'].columns:
        if col != '':
            pass
        else:
            logging.info(f"Shock {col}")
            ki_hat = pd.Series(index=sectors.index, data=1)
            li_hat = pd.Series(index=sectors.index, data=1)

            # Processing shocks
            betai_hat = {
                'sector': process_shocks(col, shocks['sector'], uniform_shock, domestic_country, psi.columns, new_consumer),
                'energy_durable': process_shocks(col, shocks['energy_durable'], uniform_shock, domestic_country, psi.columns, new_consumer),
                'nondurable_energyservices': process_shocks(col, shocks['nondurable_energyservices'], uniform_shock, domestic_country, psi.columns, new_consumer),
                'sector_IO': process_shocks(col, shocks['sector_IO'], uniform_shock, domestic_country, psi.columns, new_consumer),
                'energy_durable_IO': process_shocks(col, shocks['energy_durable_IO'], uniform_shock, domestic_country, psi.columns, new_consumer),
                'nondurable_energyservices_IO': process_shocks(col, shocks['nondurable_energyservices_IO'], uniform_shock, domestic_country, psi.columns, new_consumer)
            }

            # Create a vector full of ones for efficiency shocks
            a_efficiency = pd.Series(index=shocks['sector'].index, data=1).to_frame()  # efficiency vector
            a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
            a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
            a_efficiency.index.names = ['Sector']

            # theta: elasticity between labor/capital and intermediate inputs
            # sigma: elasticity between inputs for final demand
            # epsilon: elasticity between intermediate inputs
            # delta: elasticity between varieties of products for production
            # mu: elasticity between varieties of products for final demand
            # nu: elasticity between energy and non-energy intermediate inputs
            # kappa: elasticity between energy and durable goods
            # rho: elasticity between energy services and non-durable goods

            # Baseline calibration
            theta, sigma, epsilon, delta, mu, nu, kappa, rho = 0.5, 0.9, 0.001, 0.9, 0.9, 0.9, 0.9, 0.95
            equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi, phi, costs_energy_final,
                                                 psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable,
                                                 costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff,
                                                 Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country, descriptions,
                                                 theta, sigma, epsilon, delta, mu, nu, kappa, rho, new_consumer, share_new_consumer)
            if uniform_shock:
                uniform = '_uniform'
            else:
                uniform = ''
            equilibrium_output.to_excel(f"outputs/new_simus/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}{uniform}.xlsx")


    # # Efficiency shocks
    # ki_hat = pd.Series(index=sectors.index, inputs=1)
    # li_hat = pd.Series(index=sectors.index, inputs=1)
    # for key in betai_hat.keys():
    #     betai_hat[key] = betai_hat[key].applymap(lambda x:1)  # we only assume 1 values everywhere
    #
    # a_efficiency = pd.Series(index=betai_hat['sector'].index, inputs=1).to_frame()  # efficiency vector
    # a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
    # a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
    # a_efficiency.index.names = ['Sector']
    # a_efficiency.loc[ENERGY_SECTORS,domestic_country] = 1.5
    #
    # # Baseline calibration
    # theta, sigma, epsilon, delta, mu, nu, kappa, rho = 0.5, 0.9, 0.001, 0.9, 0.9, 0.001, 0.5, 0.95
    # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi,
    #                                      costs_energy_final,
    #                                      psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable,
    #                                      costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff,
    #                                      Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE,
    #                                      domestic_country, theta, sigma, epsilon, delta, mu, nu, kappa, rho)
    # equilibrium_output.to_excel(f"outputs/new_simus/{domestic_country}_efficiency_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}.xlsx")

    #
    # # Labor supply shocks
    # li_hat.loc[li_hat.index.get_level_values('Country') == 'EUR'] = 0.96
    # betai_hat = pd.Series(index=demand_shocks.index, inputs=1).to_frame()  # efficiency vector
    # betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: domestic_country})
    # betai_hat = betai_hat.reindex(psi.columns, axis=1, fill_value=1.0)
    # betai_hat.index.names = ['Sector']
    # betai_hat.columns.names = ['Country']
    # a_efficiency = pd.Series(index=demand_shocks.index, inputs=1).to_frame()  # efficiency vector
    # a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
    # a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
    # a_efficiency.index.names = ['Sector']
    # theta, sigma, epsilon, delta, mu, nu, kappa, rho = 0.5, 0.9, 0.001, 0.9, 0.9, 0.9, 0.9, 0.9
    # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi,
    #                                      costs_energy_final,
    #                                      psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable,
    #                                      costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff,
    #                                      Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE,
    #                                      domestic_country, theta, sigma, epsilon, delta, mu, nu, kappa, rho)
    # equilibrium_output.to_excel(f"outputs/{domestic_country}_labor_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}.xlsx")


    # list_methods = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
    # list_methods = ['hybr', 'lm', 'krylov', 'df-sane', 'linearmixing', 'diagbroyden', ]
    #
    # list_methods = ['Nelder-Mead', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    # list_methods = ['BFGS']
    # for m in list_methods:
    #     print(m)
    #     # lvec_sol = minimize(residuals_wrapper, initial_guess, jac=lambda x,*args: approx_fprime(x,residuals_wrapper,1e-8,*args), hess='2-point', method=m, options={'maxiter': 10})
    #     # lvec_sol = minimize(residuals_wrapper, initial_guess, jac=lambda x: grad_residuals(residuals_wrapper, x), method=m, options={'maxiter': 10})
    #     lvec_sol = minimize(residuals_wrapper, initial_guess, method=m, options={'maxiter': 1})
    #     print(residuals_wrapper(lvec_sol.x))

    # # Estimate the necessary adjustment in the x parameters to ensure that the mean level of the x stay constant
    # tmp = shocks['sector_IO']
    # tmp = tmp['food2']
    # tmp = tmp[tmp != 1].dropna()
    # tmp_durable = tmp[tmp.index.isin(DURABLE_GOODS)]
    # tmp_non_durable = tmp[tmp.index.isin(NON_DURABLE_GOODS)]
    # adjustment_x_nondurable = (1-(psi_non_durable.loc[tmp_non_durable.index,'EUR'] * tmp_non_durable).sum()) / (1 - psi_non_durable.loc[tmp_non_durable.index,'EUR'].sum())
    # adjustment_x_durable = (1-(psi_durable.loc[tmp_durable.index,'EUR'] * tmp_durable).sum()) / (1 - psi_durable.loc[tmp_durable.index,'EUR'].sum())