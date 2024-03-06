import pandas as pd
import numpy as np
import logging
from calibrate import CalibOutput
from scipy.optimize import fsolve, root, minimize, approx_fprime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import add_long_description, flatten_index, unflatten_index_in_df
from dataclasses import dataclass
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']
DIRTY_ENERGY_USE = ['Petro', 'FuelDist']

class OptimizationContext:
    """Class to solve the equilibrium of the model"""
    def __init__(self, li_hat,ki_hat, betai_hat, theta,sigma,epsilon,delta,mu, sectors, xsi, psi, Omega, Domestic, Delta, share_GNE, domestic_country, singlefactor):
        self.li_hat = li_hat
        self.ki_hat = ki_hat
        self.betai_hat = betai_hat
        self.theta = theta
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        self.mu = mu
        self.sectors = sectors
        self.xsi = xsi
        self.psi = psi
        self.Omega = Omega
        self.Domestic = Domestic
        self.Delta = Delta
        self.share_GNE = share_GNE
        self.domestic_country = domestic_country
        self.singlefactor = singlefactor

    def residuals_wrapper(self, lvec):
        # Call the original function but only return the first output
        res, *_ = residuals(lvec, self.li_hat, self.ki_hat, self.betai_hat, self.theta, self.sigma, self.epsilon, self.delta, self.mu, self.sectors, self.xsi, self.psi,
                            self.Omega, self.Domestic, self.Delta, self.share_GNE, singlefactor=self.singlefactor, domestic_country=self.domestic_country)
        # return (res**2).sum()
        return res

    def solve_equilibrium(self, initial_guess, method='krylov'):
        """Solves the equilibrium, using exact hat algebra. Specify the method used to find the solution (which is
        equivalent to finding the zeros of the function."""
        t1 = time.time()
        lvec_sol = root(self.residuals_wrapper, initial_guess, method=method)
        t_m = time.time() - t1
        residual = (self.residuals_wrapper(lvec_sol.x) ** 2).sum()
        logging.info(f"Method: {method:10s}, Time: {t_m:5.1f}, Residual: {residual:10.2e}")
        res, output = residuals(lvec_sol.x, self.li_hat, self.ki_hat, self.betai_hat, self.theta, self.sigma, self.epsilon, self.delta, self.mu, self.sectors, self.xsi, self.psi,
                            self.Omega, self.Domestic, self.Delta, self.share_GNE, singlefactor=self.singlefactor, domestic_country=self.domestic_country)
        return lvec_sol.x, output


@dataclass
class EquilibriumOutput:
    """Class to save the outcome of the model."""
    pi_hat: pd.DataFrame
    yi_hat: pd.DataFrame
    pi_imports_finaldemand: pd.DataFrame
    final_demand: pd.DataFrame
    domar: pd.DataFrame
    emissions_hat: pd.DataFrame
    global_variables: pd.Series
    descriptions: pd.Series

    def to_excel(self, path):
        with pd.ExcelWriter(path) as writer:
            for current_df, sheet_name in [
                (self.pi_hat, "pi_hat"),
                (self.yi_hat, "yi_hat"),
                (self.pi_imports_finaldemand, "pi_imports_finaldemand"),
                (self.final_demand, "final_demand"),
                (self.domar, "domar"),
                (self.emissions_hat, "emissions_hat"),
                (self.global_variables, "global_variables"),
                (self.descriptions, "descriptions"),
            ]:
                is_dataframe = type(current_df) is not pd.Series

                # Copy the dataframe to avoid modifying the original one
                df_to_write = current_df.copy()
                # Add long description if the index has a "Sector" level
                if "Sector" in current_df.index.names and sheet_name != "descriptions":
                    df_to_write = add_long_description(df_to_write, self.descriptions)
                # Flatten the index
                df_to_write.index = flatten_index(df_to_write.index)

                df_to_write.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    header=is_dataframe,
                    index=True)

    @classmethod
    def from_excel(cls, path):
        with pd.ExcelFile(path) as xls:
            pi_hat = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="pi_hat", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            yi_hat = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="yi_hat", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            pi_imports_finaldemand = pd.read_excel(xls, sheet_name="pi_imports_finaldemand", index_col=0).drop(columns="long_description")
            final_demand = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="final_demand", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            domar = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="domar", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"]
            )
            emissions_hat = pd.read_excel(xls, sheet_name="emissions_hat", index_col=0)
            global_variables = pd.read_excel(xls, sheet_name="global_variables", index_col=0, header=None).squeeze()
            descriptions = pd.read_excel(xls, sheet_name="descriptions", index_col=0, header=None).squeeze()
            descriptions.index.name = "Sector"
            descriptions.name = 0
        return cls(pi_hat, yi_hat, pi_imports_finaldemand, final_demand, domar, emissions_hat, global_variables, descriptions)

def residuals(lvec, li_hat,ki_hat, betai_hat,theta,sigma,epsilon,delta,mu,sectors, xsi, psi, Omega, Domestic, Delta, share_GNE, singlefactor=False, domestic_country = 'FRA'):
    """Function to compute the residuals of the model. Residuals are obtained from FOC from the model. The goal is then to
    minimize this function in order to find its zeros, corresponding to the equilibrium."""
    N = len(sectors)  # number of sectors times countries
    S = len(psi)  # number of sectors
    C = xsi.shape[1] # number of countries
    vec = np.exp(lvec)
    # vec = lvec
    pi_hat = vec[:N]
    yi_hat = vec[N:2*N]
    pi_hat = pd.Series(pi_hat, index=sectors.index)
    yi_hat = pd.Series(yi_hat, index=sectors.index)
    PSigmaY = pd.Series(index=psi.columns, data=vec[2*N:2*N+C])  #

    if singlefactor:
        w_country = pd.Series(index=psi.columns, data=vec[2*N+C:2*N+2*C])
        r_country = pd.Series(index=psi.columns, data=vec[2*N+2*C:2*N+3*C])

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

    # Price for intermediate sectors goods
    if epsilon == 1:
        price_intermediate = Omega * np.log(price_imports)
        price_intermediate = np.exp(price_intermediate.sum(axis=1))
    else:
        price_intermediate = Omega * price_imports**(1-epsilon)
        price_intermediate = price_intermediate.sum(axis=1)**(1/(1-epsilon))
    # price_intermediate = price_intermediate.to_frame()
    assert price_intermediate.shape == (N,)

    if singlefactor:  # li_hat and ki_hat are endogenous in this case
        vi_hat = np.exp(sectors['gamma'] * np.log(w_country) + (1 - sectors['gamma']) * np.log(r_country))
        hi_hat = yi_hat * (pi_hat / vi_hat) ** theta
        li_hat = hi_hat * vi_hat / w_country
        ki_hat = hi_hat * vi_hat / r_country
    else:
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

    # Price for intermediate sectors goods, final demand (price index)
    if sigma == 1:  # careful: this is actually no longer true for this specific subcase (because of the Cobb-Douglas assumption here)
        psi_new = (betai_hat * psi) / (betai_hat * psi).sum()
        price_index = betai_hat * psi * np.log(price_imports_finaldemand)
        price_index = np.exp(price_index.sum(axis=0))
    else:
        price_index = betai_hat * psi * price_imports_finaldemand**(1-sigma)
        price_index = (price_index.sum(axis=0))**(1/(1-sigma))  # sum over all sectors
    assert price_index.shape == (C,)

    # Final demand
    final_demand = (betai_hat * PSigmaY * price_index**(sigma) * price_imports_finaldemand**(mu - sigma) ).mul(pi_hat**(-mu), axis=0)

    # Intermediate demand
    intermediate_demand = (price_imports**(delta-epsilon)).mul(pi_hat**theta * yi_hat * price_intermediate**(epsilon-theta), axis=0) * pi_hat**(-delta)

    ### Residuals
    # Prices
    if theta == 1:
        res1 = np.log(pi_hat) - (sectors['eta'] * np.log(vi_hat) + (1 - sectors['eta']) * np.log(price_intermediate))
    else:
        res1 = pi_hat ** (1 - theta) - (sectors['eta'] * vi_hat ** (1 - theta) + (1 - sectors['eta']) * (price_intermediate) ** (1 - theta))

    # Quantities
    phi = sectors.loc[:, sectors.columns.str.contains('phi_')]  # share of final consumption in total output
    phi = phi.rename(columns=lambda x: x.split('_')[1])
    phi.columns.names = ['Country']  # TODO: modify in input directly
    assert phi.shape == final_demand.shape
    res2 = yi_hat - ((final_demand * phi).sum(axis=1) + (Delta * intermediate_demand).sum(axis=0))

    # World GDP
    res3 = (share_GNE * PSigmaY * price_index).sum() - 1

    revenue = sectors.loc[:, sectors.columns.str.contains('rev_')]
    rev_labor_dom = revenue['rev_labor'].xs(domestic_country, level='Country', axis=0)
    rev_capital_dom = revenue['rev_capital'].xs(domestic_country, level='Country', axis=0)
    # TODO: this line should be modified if I include more countries than just two
    li_hat_dom = li_hat.xs(domestic_country, level='Country', axis=0)
    ki_hat_dom = ki_hat.xs(domestic_country, level='Country', axis=0)

    if singlefactor:
        wi_hat_dom = w_country[domestic_country]
        ri_hat_dom = r_country[domestic_country]
        res4 = (rev_labor_dom  * li_hat_dom  * wi_hat_dom).sum() + (rev_capital_dom  * ki_hat_dom  * ri_hat_dom).sum() - (PSigmaY * price_index)[domestic_country]

        share_labor = sectors['share_labor']
        share_capital = sectors['share_capital']
        res5 = 1 - (share_labor * li_hat).groupby(level="Country").sum()
        res6 = 1 - (share_capital * ki_hat).groupby(level="Country").sum()

        res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4]), res5.to_numpy(), res6.to_numpy()])

    else:
        # Definition of GDP in other country
        wi_hat_dom = wi_hat.xs(domestic_country, level='Country', axis=0)
        ri_hat_dom = ri_hat.xs(domestic_country, level='Country', axis=0)
        res4 =(rev_labor_dom  * li_hat_dom  * wi_hat_dom).sum() + (rev_capital_dom  * ki_hat_dom  * ri_hat_dom).sum() - (PSigmaY * price_index)[domestic_country]

        res = np.concatenate([res1.to_numpy(), res2.to_numpy(), np.array([res3]), np.array([res4])])

    # budget_shares_new = final_demand.mul(pi_hat, axis=0) / (PSigmaY*price_index) * psi*xsi  # new budget shares
    # variation_welfare = (PSigmaY - 1) + (price_index - 1) + np.log((budget_shares_new*pi_hat**(sigma - 1))**(1/(1-sigma))).sum()  # TODO: il faut modifier cela car avec le nest, ce n'est plus exactement cela.
    #
    final_demand_aggregator = ((xsi * final_demand**((mu-1)/mu)).groupby('Sector').sum())**(mu/(mu-1))
    budget_shares_hat = final_demand_aggregator * price_imports_finaldemand / (PSigmaY*price_index)
    budget_shares_new = budget_shares_hat * psi
    variation_welfare = np.log(PSigmaY * price_index) + np.log(((budget_shares_new*price_imports_finaldemand**(sigma - 1)).sum())**(1/(1-sigma)))
    #
    output = {
        'pi_hat': pi_hat,
        'yi_hat': yi_hat,
        'pi_imports_finaldemand': price_imports_finaldemand,
        'li_hat': li_hat,
        'ki_hat': ki_hat,
        'PSigmaY': PSigmaY,
        'price_index': price_index,
        'GDP': PSigmaY * price_index,
        'domestic_domar': pi_hat * yi_hat / (PSigmaY * price_index),
        'domar': pi_hat * yi_hat,
        'final_demand': final_demand,
        'intermediate_demand': intermediate_demand,
        'variation_welfare': variation_welfare
    }
    if singlefactor:
        output['w_country'] = w_country
        output['r_country'] = r_country
        output['domestic_factor_labor_domar'] = li_hat * w_country / output['GDP']
        output['domestic_factor_capital_domar'] = ki_hat * r_country / output['GDP']
        output['factor_labor_domar'] = li_hat * w_country
        output['factor_capital_domar'] = ki_hat * r_country
    else:
        output['wi_hat'] = wi_hat
        output['ri_hat'] = ri_hat
        output['domestic_factor_labor_domar'] = li_hat * wi_hat / output['GDP']
        output['domestic_factor_capital_domar'] = ki_hat * ri_hat / output['GDP']
        output['factor_labor_domar'] = li_hat * wi_hat
        output['factor_capital_domar'] = ki_hat * ri_hat
    return res, output


def run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy,
                    final_use_dirty_energy, share_GNE, domestic_country, theta, sigma, epsilon, delta, mu):
    """Solves the equilibrium, under different settings."""
    singlefactor = False

    logging.info('Solving for Cobb-Douglas')
    context = OptimizationContext(li_hat, ki_hat, betai_hat, 0.99, 0.99, 0.99, 0.99, 0.99, sectors, xsi, psi, Omega,
                                  Domestic, Delta, share_GNE, domestic_country, singlefactor)
    initial_guess = np.zeros(2 * N + 2)
    sol_CD, output_CD = context.solve_equilibrium(initial_guess, method='krylov')

    logging.info('Solving for reference')
    context_ref = OptimizationContext(li_hat, ki_hat, betai_hat, theta, sigma, epsilon, delta, mu, sectors, xsi, psi,
                                      Omega, Domestic, Delta, share_GNE, domestic_country, singlefactor)
    initial_guess = np.zeros(2 * N + 2)
    sol, output_ref = context_ref.solve_equilibrium(initial_guess, method='krylov')

    logging.info('Solving for single factor')
    singlefactor = True
    context_single = OptimizationContext(li_hat, ki_hat, betai_hat, theta, sigma, epsilon, delta, mu, sectors, xsi, psi,
                                         Omega, Domestic, Delta, share_GNE, domestic_country, singlefactor)
    initial_guess = np.zeros(2 * N + 6)
    sol, output_single = context_single.solve_equilibrium(initial_guess, method='krylov')

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
    price_index = pd.concat([output_CD['price_index'].rename({i: f'price_index_{i}_CD' for i in output_CD['price_index'].index}),
                          output_ref['price_index'].rename({i: f'price_index_{i}_ref' for i in output_ref['price_index'].index}),
                          output_single['price_index'].rename({i: f'price_index_{i}_single' for i in output_single['price_index'].index})])
    variation_welfare = pd.concat([output_CD['variation_welfare'].rename({i: f'variation_welfare_{i}_CD' for i in output_CD['variation_welfare'].index}), output_ref['variation_welfare'].rename({i: f'variation_welfare_{i}_ref' for i in output_ref['variation_welfare'].index}), output_single['variation_welfare'].rename({i: f'variation_welfare_{i}_single' for i in output_single['variation_welfare'].index})])
    global_variables = pd.concat([real_GDP, price_index, variation_welfare], axis=0)

    output_dict = {
        'CD': output_CD,
        'ref': output_ref,
        'single': output_single
    }
    emissions_hat = variation_emission(output_dict, emissions, sectors_dirty_energy, final_use_dirty_energy)

    equilibrium_output = EquilibriumOutput(pi_hat, yi_hat, pi_imports_finaldemand, final_demand, domar_tot, emissions_hat, global_variables, descriptions)
    return equilibrium_output


def get_emissions_hat(yi_hat, intermediate_demand, final_demand, sectors_dirty_energy, final_use_dirty_energy):
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
    final_demand_energy = (final_demand_energy * final_use_dirty_energy).sum(
        axis=0)  # average variation of final demand of dirty energy
    # For final demand, we only consider the share of total emissions, not differentiated by fossil fuel
    total_variation_emissions = ((emissions['share_emissions_total_sectors'] * (
                variation_emissions_energy + variation_emissions_process)).groupby('Country').sum() +
                              emissions['share_emissions_total_finaldemand'].unstack('Country').sum(
                                  axis=0) * final_demand_energy)
    return total_variation_emissions


def get_emissions_total(total_variation_emissions, emissions):
    """Computes absolute variations of emissions, based on relative variations, and data on world emissions."""
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

def variation_emission(output_dict, emissions, sectors_dirty_energy, final_use_dirty_energy):
    """Estimations variation in emissions in the new equilibrium, compared to reference."""
    results = pd.DataFrame()
    total_variation_emissions_io = input_output_calculation(betai_hat, Leontieff, Gamma, sectors, sectors_dirty_energy, final_use_dirty_energy)
    total_variation_emissions_io = total_variation_emissions_io.to_frame().rename(columns={0: f'emissions_IO'})
    absolute_emissions_io = get_emissions_total(total_variation_emissions_io, emissions)
    total_variation_emissions_io = pd.concat([total_variation_emissions_io, absolute_emissions_io.to_frame().rename(columns={0: f'emissions_IO'})], axis=0)
    results = pd.concat([results, total_variation_emissions_io], axis=1)
    for key, output in output_dict.items():
        intermediate_demand = output['intermediate_demand']
        yi_hat = output['yi_hat']
        final_demand = output['final_demand']
        total_variation_emissions = get_emissions_hat(yi_hat, intermediate_demand, final_demand, sectors_dirty_energy, final_use_dirty_energy)
        total_variation_emissions = total_variation_emissions.to_frame().rename(columns={0: f'emissions_{key}'})
        absolute_emissions = get_emissions_total(total_variation_emissions, emissions)
        total_variation_emissions = pd.concat([total_variation_emissions, absolute_emissions.to_frame().rename(columns={0: f'emissions_{key}'})], axis=0)

        results = pd.concat([results, total_variation_emissions], axis=1)
    return results


def process_output(dict_paths):
    """Creates an output file which contains the variation of emissions for each country and each sector, in correct format.
    Function is used for plots only."""
    emissions_dict = dict()
    emissions_absolute_dict = dict()
    welfare_dict = dict()
    for k in dict_paths.keys():
        equilibrium_output = EquilibriumOutput.from_excel(dict_paths[k])
        emissions_dict[k] = equilibrium_output.emissions_hat
        welfare_dict[k] = equilibrium_output.global_variables

    # Get emissions variation
    concatenated_dfs = []
    concatenated_dfs_2 = []

    def rename_index(L, country):
        new_index = []
        for i in L:
            tmp = i.split('_absolute')[0]
            if tmp == country:
                new_index.append('Dom.')
            else:
                new_index.append('RoW')
        return new_index

    for key, df in emissions_dict.items():
        country = df.index[0]
        c, sector = key.split(' - ')[0], key.split(' - ')[1]
        transformed_df = df.loc[country] - 1
        transformed_df = transformed_df.to_frame().T
        transformed_df.index = pd.MultiIndex.from_product([transformed_df.index, [sector]], names=['Country', 'Sector'])
        concatenated_dfs.append(transformed_df)

        emissions_absolute = df.iloc[-2:,:]

        emissions_absolute.index = rename_index(emissions_absolute.index, country)
        emissions_absolute.index = pd.MultiIndex.from_product([[country], [sector], emissions_absolute.index], names=['Country', 'Sector', 'Category'])
        concatenated_dfs_2.append(emissions_absolute)
    emissions_df = pd.concat(concatenated_dfs, axis=0)
    emissions_df = emissions_df.rename(columns={
        'emissions_CD': 'D+CD',
        'emissions_ref': 'D+CD+CES',
        'emissions_single': 'D'
    })
    emissions_df = emissions_df.reindex(['D', 'D+CD', 'D+CD+CES'], axis=1)
    emissions_df.columns.names = ['Category']
    emissions_df = emissions_df.stack()

    emissions_absolute_df = pd.concat(concatenated_dfs_2, axis=0)
    emissions_absolute_df = emissions_absolute_df.rename(columns={
        'emissions_CD': 'D+CD',
        'emissions_ref': 'D+CD+CES',
        'emissions_single': 'D'
    })
    emissions_absolute_df = emissions_absolute_df.reindex(['D', 'D+CD', 'D+CD+CES'], axis=1)
    emissions_absolute_df = emissions_absolute_df.stack()
    # rename last level of index from None to Effect
    # the level is the last level
    emissions_absolute_df.index = emissions_absolute_df.index.set_names('Effect', level=-1)


    # Get welfare variation
    concatenated_dfs = []
    for key, df in welfare_dict.items():
        welfare = df[df.index.str.contains('welfare')]
        welfare.index = pd.MultiIndex.from_tuples((i.split('_')[-2], i.split('_')[-1]) for i in welfare.index)
        sector = key.split(' - ')[1]
        # add a level to the index, which is always equal to sector
        # current index of welfare is already multilevel
        # we add a level to the index, which is always equal to sector
        welfare.index = pd.MultiIndex.from_tuples((sector, i[0], i[1]) for i in welfare.index)
        welfare.index.names=['Country', 'Sector', 'Category']
        welfare = welfare.unstack()
        concatenated_dfs.append(welfare)

    welfare_df = pd.concat(concatenated_dfs, axis=0)
    welfare_df = welfare_df.rename(columns={
        'CD': 'D+CD',
        'ref': 'D+CD+CES',
        'single': 'D'
    })
    welfare_df = welfare_df.reindex(['D', 'D+CD', 'D+CD+CES'], axis=1)

    with pd.ExcelWriter(Path('outputs/welfare.xlsx')) as writer:
        welfare_df.to_excel(
            writer,
            header=True,
            index=True)

    with pd.ExcelWriter(Path('outputs/emissions.xlsx')) as writer:
        emissions_df.to_excel(
            writer,
            header=True,
            index=True)

    with pd.ExcelWriter(Path('outputs/emissions_absolute_df.xlsx')) as writer:
        emissions_absolute_df.to_excel(
            writer,
            header=True,
            index=True)
    return emissions_df, emissions_absolute_df, welfare_df


def input_output_calculation(betai_hat, Leontieff, Gamma, sectors, sectors_dirty_energy, final_use_dirty_energy):
    """Takes vector of shocks and outputs estimated variation in emissions from simple input output framework.
    This framework does not include any price effect, and only relies on Leontieff matrix and direct input coefficients
    matrix."""
    # We calculate share of production that stems from final demand
    phi = sectors.loc[:, sectors.columns.str.contains('phi_')]  # share of final consumption in total output
    phi = phi.rename(columns=lambda x: x.split('_')[1])
    phi.columns.names = ['Country']
    pyi = sectors.loc[:, 'pyi']
    final_use_init = phi.mul(pyi, axis=0)  # final use expenditures in the initial state

    final_use_new = betai_hat.copy() * final_use_init  # we calculate new final use

    pyi_new = Leontieff.mul(final_use_new.sum(axis=1), axis=0).sum(axis=0)  # we calculate required production from the Leontieff accounting equation y = L^T b
    yi_hat = pyi_new / pyi  # we calculate the relative variation for y, assuming there are no price changes in this setting

    final_demand = betai_hat.copy()
    intermediate_demand_hat = pd.concat([yi_hat]*len(Leontieff), axis=1)
    intermediate_demand_hat.columns = Gamma.columns
    variation_emissions = get_emissions_hat(yi_hat, intermediate_demand_hat, final_demand, sectors_dirty_energy, final_use_dirty_energy)
    return variation_emissions


if __name__ == '__main__':

    code_country = {
        'france': 'FRA',
        'united_states_of_america': 'USA',
        'europe': 'EUR'
    }
    country = 'europe'
    domestic_country = code_country[country]
    filename = f"outputs/calib_{country}.xlsx"
    fileshocks = "data_deep/shocks_demand_06032024.xlsx"

    calib = CalibOutput.from_excel(filename)
    sectors, emissions, xsi, psi, Omega, Gamma, Leontieff, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, descriptions = calib.sectors, calib.emissions, calib.xsi, calib.psi, calib.Omega, calib.Gamma, calib.Leontieff, calib.Domestic, calib.Delta, calib.sectors_dirty_energy, calib.final_use_dirty_energy, calib.share_GNE, calib.descriptions
    N = len(sectors)

    # sectors['gamma'] = 1.0
    demand_shocks = pd.read_excel(fileshocks, index_col=0, header=0)

    for col in demand_shocks.columns:
        logging.info(f"Shock {col}")
        ki_hat = pd.Series(index=sectors.index, data=1)
        li_hat = pd.Series(index=sectors.index, data=1)

        betai_hat = demand_shocks[col]  # get specific shocks
        betai_hat = betai_hat.to_frame()
        betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: domestic_country})

        # Preferences shocks are shared across countries
        # betai_hat = pd.concat([betai_hat]*len(psi.columns), axis=1)
        # betai_hat.columns = psi.columns

        # Preferences shocks are specific to domestic country
        betai_hat = betai_hat.reindex(psi.columns, axis=1, fill_value=1.0)

        betai_hat.index.names = ['Sector']
        betai_hat.columns.names = ['Country']

        # list_methods = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
        # list_methods = ['hybr', 'lm', 'krylov', 'df-sane', 'linearmixing', 'diagbroyden', ]

        # Baseline calibration
        theta, sigma, epsilon, delta, mu = 0.5, 0.9, 0.001, 0.9, 0.9
        equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country,
                                            theta, sigma, epsilon, delta, mu)
        equilibrium_output.to_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}.xlsx")
        #
        #
        # # Substitution with imports calibration for production
        # theta, sigma, epsilon, delta, mu = 0.5, 0.9, 0.001, 5, 0.9
        # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country,
        #                                     theta, sigma, epsilon, delta, mu)
        # equilibrium_output.to_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}.xlsx")

        # # Substitution with imports calibration for final demand
        # theta, sigma, epsilon, delta, mu = 0.5, 0.9, 0.001, 0.9, 5
        # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country,
        #                                     theta, sigma, epsilon, delta, mu)
        # equilibrium_output.to_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}.xlsx")

        #
        # # # Low elasticity calibration
        # # theta, sigma, epsilon, delta, mu = 0.3, 0.7, 0.001, 0.9, 0.9
        # # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country,
        # #                                     theta, sigma, epsilon, delta, mu)
        # # equilibrium_output.to_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}.xlsx")
        # #
        # # High calibration
        # theta, sigma, epsilon, delta, mu = 0.9, 0.9, 0.9, 0.9, 0.9
        # equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, sectors, emissions, xsi, psi, Omega, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, domestic_country,
        #                                     theta, sigma, epsilon, delta, mu)
        # equilibrium_output.to_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_all.xlsx")

        # eq2 = EquilibriumOutput.from_excel(f"outputs/{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}.xlsx")

        # I want to multiply domar['domestic_factor_labor_domar_hat'].groupby('Country') with sectors['rev_labor'].groupby('Country') and sum over sectors


d = {
    'EUR - Food': Path('outputs/EUR_food_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
    'FRA - Food': Path('outputs/FRA_food_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
    'EUR - Mobility': Path('outputs/EUR_distance_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
    'FRA - Mobility': Path('outputs/FRA_distance_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx')
}

# d = {
#     'EUR - Food': Path('outputs/EUR_all_food_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
#     'FRA - Food': Path('outputs/FRA_all_food_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
#     'EUR - Mobility': Path('outputs/EUR_all_distance_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx'),
#     'FRA - Mobility': Path('outputs/FRA_all_distance_theta0.5_sigma0.9_epsilon0.001_delta0.9_mu0.9.xlsx')
# }

emissions_df, emissions_absolute_df, welfare_df = process_output(d)





    # list_methods = ['Nelder-Mead', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    # list_methods = ['BFGS']
    # for m in list_methods:
    #     print(m)
    #     # lvec_sol = minimize(residuals_wrapper, initial_guess, jac=lambda x,*args: approx_fprime(x,residuals_wrapper,1e-8,*args), hess='2-point', method=m, options={'maxiter': 10})
    #     # lvec_sol = minimize(residuals_wrapper, initial_guess, jac=lambda x: grad_residuals(residuals_wrapper, x), method=m, options={'maxiter': 10})
    #     lvec_sol = minimize(residuals_wrapper, initial_guess, method=m, options={'maxiter': 1})
    #     print(residuals_wrapper(lvec_sol.x))