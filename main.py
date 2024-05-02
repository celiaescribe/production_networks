import os

from exercice_hat_fun_nicefigure import CalibOutput, read_file_shocks, process_shocks, run_equilibrium

import logging
import pandas as pd
import datetime
from pathlib import Path

CODE_COUNTRY = {
    'france': 'FRA',
    'united_states_of_america': 'USA',
    'europe': 'EUR'
}

DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']
DIRTY_ENERGY_USE = ['Petro', 'FuelDist']

ENERGY_SECTORS = DIRTY_ENERGY_SECTORS +  ['Power']


if __name__ == '__main__':

    country = 'europe'
    fileshocks = "data_deep/shocks_demand_09042024.xlsx"
    uniform_shock = False
    new_consumer = False
    share_new_consumer = 0.5
    efficiency = True

    domestic_country = CODE_COUNTRY[country]
    filename = f"outputs/calib_{country}.xlsx"

    folder_date = datetime.datetime.now().strftime("%Y%m%d")
    folder_to_save = Path(f'outputs/simulations_{folder_date}')
    if not folder_to_save.is_dir():
        os.mkdir(folder_to_save)


    calib = CalibOutput.from_excel(filename)
    if new_consumer:
        calib.add_final_consumer(country=domestic_country, share_new_consumer=share_new_consumer)
    sectors, emissions, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, descriptions = calib.sectors, calib.emissions, calib.xsi, calib.psi, calib.phi, calib.costs_energy_final, calib.psi_energy, calib.psi_non_energy, calib.costs_durable_final, calib.psi_durable, calib.psi_non_durable, calib.costs_energy_services_final, calib.Omega, calib.costs_energy, calib.Omega_energy, calib.Omega_non_energy ,calib.Gamma, calib.Leontieff, calib.Domestic, calib.Delta, calib.sectors_dirty_energy, calib.final_use_dirty_energy, calib.share_GNE, calib.descriptions
    N = len(sectors)

    shocks = read_file_shocks(fileshocks)

    if not efficiency:
        list_elasticities = [0.001, 0.5, 0.9]

        for elasticity in list_elasticities:

            for col in shocks['sector'].columns:
                if col == '':
                    pass
                else:
                    logging.info(f"Shock {col}")
                    ki_hat = pd.Series(index=sectors.index, data=1)
                    li_hat = pd.Series(index=sectors.index, data=1)

                    betai_hat = process_shocks(col, shocks, uniform_shock, domestic_country, psi.columns, new_consumer)

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
                    theta, sigma, epsilon, delta, mu, nu, kappa, rho = 0.5, 0.9, 0.001, 0.9, 0.9, elasticity, 0.5, 0.9
                    equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi,
                                                         phi, costs_energy_final,
                                                         psi_energy, psi_non_energy, costs_durable_final, psi_durable,
                                                         psi_non_durable,
                                                         costs_energy_services_final, Omega, costs_energy, Omega_energy,
                                                         Omega_non_energy, Gamma, Leontieff,
                                                         Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy,
                                                         share_GNE, domestic_country, descriptions,
                                                         theta, sigma, epsilon, delta, mu, nu, kappa, rho, new_consumer,
                                                         share_new_consumer)
                    if uniform_shock:
                        uniform = '_uniform'
                    else:
                        uniform = ''
                    equilibrium_output.to_excel(Path(folder_to_save) / Path(f"{domestic_country}_{col}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}{uniform}.xlsx"))

    else:  # we study efficiency rebound effects
        fileshocks = "data_deep/shocks_demand_09042024_efficiency.xlsx"
        shocks = read_file_shocks(fileshocks)
        list_elasticities = [0.1, 0.5, 0.9]

        for elasticity in list_elasticities:

            ki_hat = pd.Series(index=sectors.index, data=1)
            li_hat = pd.Series(index=sectors.index, data=1)
            col = 'efficiency'
            betai_hat = process_shocks(col, shocks, uniform_shock, domestic_country, psi.columns, new_consumer)


            a_efficiency = pd.Series(index=betai_hat['sector'].index, data=1).to_frame()  # efficiency vector
            a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
            a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
            a_efficiency.index.names = ['Sector']
            a_efficiency.loc[ENERGY_SECTORS,domestic_country] = 1.069  # to match the same IO effect as in the other shock scenarios

            # Baseline calibration
            theta, sigma, epsilon, delta, mu, nu, kappa, rho = 0.5, 0.9, 0.001, 0.9, 0.9, 0.001, 0.5, elasticity
            equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions, xsi, psi,
                                                 phi, costs_energy_final,
                                                 psi_energy, psi_non_energy, costs_durable_final, psi_durable,
                                                 psi_non_durable,
                                                 costs_energy_services_final, Omega, costs_energy, Omega_energy,
                                                 Omega_non_energy, Gamma, Leontieff,
                                                 Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy,
                                                 share_GNE, domestic_country, descriptions,
                                                 theta, sigma, epsilon, delta, mu, nu, kappa, rho, new_consumer,
                                                 share_new_consumer)
            equilibrium_output.to_excel(Path(folder_to_save) / Path(f"{domestic_country}_efficiency_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}.xlsx"))
