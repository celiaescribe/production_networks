import os
from copy import deepcopy

import pandas as pd
import numpy as np
import logging
from calibrate import CalibOutput
from exercice_hat_fun_nicefigure import read_file_shocks, variation_emission, process_shocks
from scipy.optimize import fsolve, root, minimize, approx_fprime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import EquilibriumOutput, add_long_description, flatten_index, unflatten_index_in_df, get_save_path, load_config
from dataclasses import dataclass
import time
from pathlib import Path
import datetime
import argparse

CODE_COUNTRY = {
    'france': 'FRA',
    'united_states_of_america': 'USA',
    'europe': 'EUR'
}

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

class OptimizationContextLabor:
    """Class to solve the equilibrium of the model"""
    def __init__(self,ki_hat, betai_hat, alpha_hat, a_efficiency, theta,sigma,epsilon,delta, mu, nu, kappa, rho, epsilon_frisch, sectors, xsi, psi, phi, psi_energy,
                 costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy,
                 Omega_non_energy, Domestic, Delta, share_GNE, domestic_country):
        self.ki_hat = ki_hat
        self.betai_hat = betai_hat
        self.alpha_hat = alpha_hat
        self.a_efficiency = a_efficiency
        self.theta = theta
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.rho = rho
        self.epsilon_frisch = epsilon_frisch
        self.sectors = sectors
        self.xsi = xsi
        self.psi = psi
        self.phi = phi
        self.psi_energy = psi_energy
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

    def residuals_wrapper(self, lvec):
        # Call the original function but only return the first output
        res, *_ = residuals_labor(lvec, self.ki_hat, self.betai_hat, self.alpha_hat, self.a_efficiency, self.theta, self.sigma, self.epsilon, self.delta,
                            self.mu, self.nu, self.kappa, self.rho, self.epsilon_frisch, self.sectors, self.xsi, self.psi, self.phi, self.psi_energy,
                            self.costs_durable_final, self.psi_durable, self.psi_non_durable, self.costs_energy_services_final, self.Omega, self.costs_energy,
                            self.Omega_energy, self.Omega_non_energy, self.Domestic, self.Delta,
                            self.share_GNE, domestic_country=self.domestic_country)
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
        res, output = residuals_labor(lvec_sol.x, self.ki_hat, self.betai_hat, self.alpha_hat, self.a_efficiency, self.theta, self.sigma, self.epsilon,
                                self.delta, self.mu, self.nu, self.kappa, self.rho, self.epsilon_frisch, self.sectors, self.xsi, self.psi, self.phi, self.psi_energy,
                                self.costs_durable_final, self.psi_durable, self.psi_non_durable, self.costs_energy_services_final, self.Omega, self.costs_energy,
                                self.Omega_energy, self.Omega_non_energy, self.Domestic, self.Delta,
                                self.share_GNE, domestic_country=self.domestic_country)
        return lvec_sol.x, output

    def find_root(self, initial_guess):
        # return root(self.residuals_wrapper, initial_guess, method='krylov')
        for method in ['krylov', 'hybr', 'lm']:
            try:
                options = {'disp': True} if method in ['krylov', 'hybr'] else {}
                lvec_sol = root(self.residuals_wrapper, initial_guess, method=method)
                return lvec_sol, method
            except Exception as e:
                logging.info(f"Method {method} failed with an unexpected error: {e}. Trying next method...")
        raise ValueError("All methods failed.")


def residuals_labor(lvec, ki_hat, betai_hat, alpha_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho, epsilon_frisch, sectors, xsi, psi, phi, psi_energy,
                    costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                    Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country = 'FRA'):
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
    N = len(sectors)  # number of sectors times countries
    S = len(psi)  # number of sectors
    C = xsi.shape[1]
    C_consumer = C
    vec = np.exp(lvec)
    pi_hat = vec[:N]
    yi_hat = vec[N:2*N]
    # li_hat = vec[2*N:3*N]
    pi_hat = pd.Series(pi_hat, index=sectors.index)
    yi_hat = pd.Series(yi_hat, index=sectors.index)
    # li_hat = pd.Series(li_hat, index=sectors.index)
    # PSigmaY = pd.Series(index=psi.columns, data=vec[3*N:3*N+C_consumer])
    PSigmaY = pd.Series(index=psi.columns, data=vec[2*N:2*N+C_consumer])
    w_country = pd.Series(index=psi.columns, data=vec[2 * N + C_consumer:2 * N + C_consumer + C])

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

    # we simplify, and get rid of the capital/labor nest. Productive sectors only rely on labor
    # wi_hat = pi_hat * (yi_hat / li_hat)**(1/theta)
    # vi_hat = wi_hat

    vi_hat = w_country
    hi_hat = yi_hat * (pi_hat / vi_hat) ** theta
    li_hat = hi_hat * vi_hat / w_country

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

    # World GDP is the numeraire, and stays the same
    res3 = (share_GNE * nominal_GDP).sum() - 1


    revenue = sectors.loc[:, sectors.columns.str.contains('rev_')]
    rev_labor_dom = revenue['rev_labor'].xs(domestic_country, level='Country', axis=0)
    rev_capital_dom = revenue['rev_capital'].xs(domestic_country, level='Country', axis=0)
    # TODO: this line should be modified if I include more countries than just two
    li_hat_dom = li_hat.xs(domestic_country, level='Country', axis=0)
    # ki_hat_dom = ki_hat.xs(domestic_country, level='Country', axis=0)


    # # Conditions from having endogenous labor
    # # Tentative with labor supply and wages all equal
    # share_labor = sectors['share_labor']
    # wi_hat_dom = wi_hat.xs(domestic_country, level='Country', axis=0)
    # # ri_hat_dom = ri_hat.xs(domestic_country, level='Country', axis=0)
    # res4 =((rev_labor_dom+rev_capital_dom)  * li_hat_dom  * wi_hat_dom).sum() - (nominal_GDP)[domestic_country]
    #
    # res5 = wi_hat_dom[1:] - wi_hat_dom[0]  # wages are the same across sectors in each country
    # res6 = wi_hat.xs('ROW', level='Country', axis=0)[1:] - wi_hat.xs('ROW', level='Country', axis=0)[0]
    # # res7 = (epsilon_frisch + 1) / epsilon_frisch * (li_hat * share_labor).groupby(level="Country").sum() - (alpha_hat + 1/epsilon_frisch)
    # res7 = ((li_hat * share_labor).groupby(level="Country").sum())**((epsilon_frisch + 1) / epsilon_frisch) - alpha_hat
    # # res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res5.to_numpy(), res6.to_numpy(), res7.to_numpy()])

    # Condition by defining all wages
    # res7 = ((li_hat * share_labor).groupby(level="Country").sum()) ** (1 / epsilon_frisch) - alpha_hat * wi_hat / nominal_GDP
    # res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res7.to_numpy()])

    # Tentative with other definition of wage
    # res7 = - wi_hat - alpha_hat * ((li_hat * share_labor).groupby(level="Country").sum()) ** (1/epsilon_frisch)
    # res7 = wi_hat - alpha_hat / ((li_hat * share_labor).groupby(level="Country").sum())   # tentative other formulation for utility from labor
    # res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res7.to_numpy()])

    # # Tentative with leisure
    # res7 = alpha_hat * leisure + wi_hat
    # res8 = 1 - leisure * 0.3 - 0.7 * ((li_hat * share_labor).groupby(level="Country").sum())
    # res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res7.to_numpy(), res8.to_numpy()])

    wi_hat_dom = w_country[domestic_country]
    res4 = ((rev_labor_dom+rev_capital_dom)  * li_hat_dom  * wi_hat_dom).sum() - (nominal_GDP)[domestic_country]

    share_labor = sectors['share_labor']
    res5 = 1/alpha_hat - (share_labor * li_hat).groupby(level="Country").sum()
    # res5 = alpha_hat - ((li_hat * share_labor).groupby(level="Country").sum())**((epsilon_frisch + 1) / epsilon_frisch)

    res = np.concatenate(
        [res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res5.to_numpy()])

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
        'variation_welfare': variation_welfare,
        'w_country' : w_country,
        'domestic_factor_labor_domar': li_hat * w_country / nominal_GDP,
        'factor_labor_domar': li_hat * w_country
    }
    return res, output


def run_equilibrium_labor(ki_hat, betai_hat, alpha_hat, a_efficiency, sectors, emissions, xsi, psi, phi, psi_energy,
                    costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy,
                    Gamma, Leontieff, Ghosh, Domestic, Delta, sectors_dirty_energy,
                    final_use_dirty_energy, share_GNE, domestic_country, descriptions, reference_parameters):
    """Solves the equilibrium, under different settings."""

    C = xsi.shape[1]
    C_consumer = C

    theta, sigma, epsilon, delta, mu, nu, kappa, rho, epsilon_frisch = reference_parameters['theta'], reference_parameters['sigma'], reference_parameters['epsilon'], reference_parameters['delta'], reference_parameters['mu'], reference_parameters['nu'], reference_parameters['kappa'], reference_parameters['rho'], reference_parameters['epsilon_frisch']

    N = len(sectors)

    singlefactor = 'specific'

    logging.info('Solving for reference')
    context_ref = OptimizationContextLabor(ki_hat, betai_hat, alpha_hat, a_efficiency, theta, sigma, epsilon, delta, mu, nu, kappa, rho, epsilon_frisch,
                                      sectors, xsi, psi, phi, psi_energy, costs_durable_final,
                                      psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                                      Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country)
    initial_guess = np.zeros(3 * N + C_consumer)
    initial_guess = np.zeros(2 * N + C_consumer + 2)
    sol, output_ref = context_ref.solve_equilibrium(initial_guess)

    logging.info('Solving for Cobb-Douglas')
    elasticity_cb = 0.97
    context = OptimizationContextLabor(ki_hat, betai_hat, alpha_hat, a_efficiency, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb, elasticity_cb,
                                       elasticity_cb, elasticity_cb, elasticity_cb, epsilon_frisch,
                                      sectors, xsi, psi, phi, psi_energy, costs_durable_final,
                                      psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy,
                                      Omega_energy, Omega_non_energy, Domestic, Delta, share_GNE, domestic_country)
    initial_guess = np.zeros(3 * N + C_consumer)
    initial_guess = np.zeros(2 * N + C_consumer + 2)
    sol_CD, output_CD = context.solve_equilibrium(initial_guess)



    output_dict = {
        'CD': output_CD,
        'ref': output_ref
    }

    pi_hat = pd.concat([df['pi_hat'].rename(f"price_hat_{suffix}") for suffix, df in output_dict.items()], axis=1)
    yi_hat = pd.concat([df['yi_hat'].rename(f"yi_hat_{suffix}") for suffix, df in output_dict.items()], axis=1)
    final_demand = pd.concat([df['final_demand'].rename(columns={col: f'{col}_{suffix}' for col in df['final_demand'].columns}) for suffix, df in output_dict.items()], axis=1)
    pi_imports_finaldemand = pd.concat([df['pi_imports_finaldemand'].rename(columns={col: f'{col}_{suffix}' for col in df['pi_imports_finaldemand'].columns}) for suffix, df in output_dict.items()], axis=1)
    domestic_domar = pd.concat([df['domestic_domar'].rename(f"domestic_domar_hat_{suffix}") for suffix, df in output_dict.items()], axis=1)
    domar = pd.concat([df['domar'].rename(f"domar_hat_{suffix}") for suffix, df in output_dict.items()], axis=1)
    domestic_factor_labor_domar = pd.concat([output_CD['domestic_factor_labor_domar'].to_frame(name='domestic_factor_labor_domar_CD_hat'), output_ref['domestic_factor_labor_domar'].to_frame(name='domestic_factor_labor_domar_hat')], axis=1)
    factor_labor_domar = pd.concat([output_CD['factor_labor_domar'].to_frame(name='factor_labor_domar_CD_hat'), output_ref['factor_labor_domar'].to_frame(name='factor_labor_domar_hat')], axis=1)
    domar_tot = pd.concat([domestic_domar, domar, domestic_factor_labor_domar, factor_labor_domar], axis=1)
    real_GDP = pd.concat([output_CD['PSigmaY'].rename({i: f'real_GDP_{i}_CD' for i in output_CD['PSigmaY'].index}), output_ref['PSigmaY'].rename({i: f'real_GDP_{i}_ref' for i in output_ref['PSigmaY'].index})])
    GDP = pd.concat([output_CD['GDP'].rename({i: f'GDP_{i}_CD' for i in output_CD['GDP'].index}), output_ref['GDP'].rename({i: f'GDP_{i}_ref' for i in output_ref['GDP'].index})])
    price_list_finaldemand = ['w_country', 'price_index', 'price_index_energy', 'price_index_durable', 'price_index_non_durable', 'price_index_energy_services',
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
    }  # variation in welfare and capital across sectors
    labor_capital = pd.concat([labor_capital_info[key] for key in labor_capital_info.keys()], axis=1)
    variation_welfare = pd.concat([output_CD['variation_welfare'].rename({i: f'variation_welfare_{i}_CD' for i in output_CD['variation_welfare'].index}), output_ref['variation_welfare'].rename({i: f'variation_welfare_{i}_ref' for i in output_ref['variation_welfare'].index})])
    global_variables = pd.concat([GDP, real_GDP, price_index_finaldemand, variation_welfare], axis=0)

    emissions_hat, emissions_detail = variation_emission(output_dict, betai_hat, emissions, Leontieff, Gamma, phi, xsi, sectors, sectors_dirty_energy, final_use_dirty_energy, False, domestic_country, 0)  # TODO: à modifier avec les betaihat

    equilibrium_output = EquilibriumOutput(pi_hat, yi_hat, price_production, pi_imports_finaldemand, final_demand, output_ref['intermediate_demand'], domar_tot, labor_capital, emissions_hat, emissions_detail, global_variables, descriptions)
    return equilibrium_output


def run_simulation_labor(config):
    """Run the simulation with the specified configuration"""
    country = config['country']
    domestic_country = CODE_COUNTRY[country]
    filename = f"outputs/calib_{country}.xlsx"

    folder_date = datetime.datetime.now().strftime("%Y%m%d")
    folder_to_save = Path(f'outputs/simulations_{folder_date}')
    if not folder_to_save.is_dir():
        os.mkdir(folder_to_save)

    calib = CalibOutput.from_excel(filename)
    if config['new_consumer']['activated']:
        calib.add_final_consumer(country=domestic_country)

    sectors, emissions, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Ghosh, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, descriptions = calib.sectors, calib.emissions, calib.xsi, calib.psi, calib.phi, calib.costs_energy_final, calib.psi_energy, calib.psi_non_energy, calib.costs_durable_final, calib.psi_durable, calib.psi_non_durable, calib.costs_energy_services_final, calib.Omega, calib.costs_energy, calib.Omega_energy, calib.Omega_non_energy, calib.Gamma, calib.Leontieff, calib.Ghosh, calib.Domestic, calib.Delta, calib.sectors_dirty_energy, calib.final_use_dirty_energy, calib.share_GNE, calib.descriptions

    shocks = read_file_shocks(config['fileshocks'])

    alpha_hat = pd.Series({'EUR': config['alpha_shock']['EUR'], 'ROW': config['alpha_shock']['ROW']})
    alpha_hat.index.names = ['Country']
    if config['uniform_shock']:  # we apply a uniform shock
        alpha_hat['ROW'] = alpha_hat['EUR']

    # Baseline calibration
    # theta: elasticity between labor/capital and intermediate inputs
    # sigma: elasticity between inputs for final demand
    # epsilon: elasticity between intermediate inputs
    # delta: elasticity between varieties of products for production
    # mu: elasticity between varieties of products for final demand
    # nu: elasticity between energy and non-energy intermediate inputs
    # kappa: elasticity between energy and durable goods
    # rho: elasticity between energy services and non-durable goods
    reference_parameters = config['reference_parameters']

    if config['sensitivity']['activated']:  # we do a sensitivity analysis
        for param in config['sensitivity']['parameter'].keys():
            for elasticity in config['sensitivity']['parameter'][param]:
                for s in config['shocks']:
                    assert s in shocks['sector'].columns, "List of shocks is not correctly specified"
                    if s == '':
                        pass
                    else:
                        logging.info(f"Shock {s}")
                        ki_hat = pd.Series(index=sectors.index, data=1)

                        betai_hat = process_shocks(s, shocks, config['uniform_shock'], domestic_country, psi.columns, config['new_consumer']['activated'])

                        # Create a vector full of ones for efficiency shocks
                        a_efficiency = pd.Series(index=betai_hat['sector'].index, data=1).to_frame()  # efficiency vector
                        a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
                        a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
                        a_efficiency.index.names = ['Sector']

                        if config['efficiency']['activated']:
                            a_efficiency.loc[ENERGY_SECTORS, domestic_country] = config['efficiency']['value']  # to match the same IO effect as in the other shock scenarios

                        reference_parameters[param] = elasticity

                        equilibrium_output = run_equilibrium_labor(ki_hat, betai_hat, alpha_hat, a_efficiency,
                                                                   sectors, emissions, xsi, psi,
                                                                   phi, psi_energy, costs_durable_final, psi_durable,
                                                             psi_non_durable,
                                                             costs_energy_services_final, Omega, costs_energy, Omega_energy,
                                                             Omega_non_energy, Gamma, Leontieff, Ghosh,
                                                             Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy,
                                                             share_GNE, domestic_country, descriptions,
                                                             reference_parameters)

                        save_path = get_save_path(reference_parameters, domestic_country, s, config['uniform_shock'], new_consumer=config['new_consumer']['activated'], labor=True)
                        equilibrium_output.to_excel(Path(folder_to_save) / Path(save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run sufficiency production network simulations.")
    parser.add_argument('--config', type=str, default='config_labor.json', help="Path to the json configuration file.")
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    run_simulation_labor(config)