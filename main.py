import os
from copy import deepcopy

from calibrate import add_final_consumer_share
from evaluate_shock import CalibOutput, read_file_shocks, process_shocks, run_equilibrium
from utils import load_config, get_save_path
import logging
import pandas as pd
import datetime
from pathlib import Path
import argparse
import json

CODE_COUNTRY = {
    'france': 'FRA',
    'united_states_of_america': 'USA',
    'europe': 'EUR',
    'eu': 'EUR'
}

DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']
DIRTY_ENERGY_USE = ['Petro', 'FuelDist']

ENERGY_SECTORS = DIRTY_ENERGY_SECTORS +  ['Power']

def run_simulation(config):
    """Run the simulation with the specified configuration"""
    country = config['country']
    year = config['year']
    domestic_country = CODE_COUNTRY[country]
    filename = f"outputs/calib_{country}_{year}.xlsx"

    folder_date = datetime.datetime.now().strftime("%Y%m%d")
    folder_to_save = Path(f'outputs/simulations_{folder_date}')
    if not folder_to_save.is_dir():
        os.mkdir(folder_to_save)

    with open(os.path.join(folder_to_save, 'config.json'), "w") as outfile:
        outfile.write(json.dumps(config, indent=4))

    calib = CalibOutput.from_excel(filename)
    if config['new_consumer']['activated']:
        calib.add_final_consumer(country=domestic_country)

    sectors, emissions, xsi, psi, phi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, costs_energy_services_final, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy, share_GNE, descriptions = calib.sectors, calib.emissions, calib.xsi, calib.psi, calib.phi, calib.costs_energy_final, calib.psi_energy, calib.psi_non_energy, calib.costs_durable_final, calib.psi_durable, calib.psi_non_durable, calib.costs_energy_services_final, calib.Omega, calib.costs_energy, calib.Omega_energy, calib.Omega_non_energy, calib.Gamma, calib.Leontieff, calib.Domestic, calib.Delta, calib.sectors_dirty_energy, calib.final_use_dirty_energy, calib.share_GNE, calib.descriptions

    shocks = read_file_shocks(config['fileshocks'])

    # Baseline calibration
    # theta: elasticity between labor/capital and intermediate inputs
    # sigma: elasticity between inputs for final demand
    # epsilon: elasticity between intermediate inputs
    # delta: elasticity between varieties of products for production
    # mu: elasticity between varieties of products for final demand
    # nu: elasticity between energy and non-energy intermediate inputs
    # kappa: elasticity between energy and durable goods
    # rho: elasticity between energy services and non-durable goods

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
                        li_hat = pd.Series(index=sectors.index, data=1)

                        betai_hat = process_shocks(s, shocks, config['uniform_shock'], domestic_country, psi.columns, config['new_consumer']['activated'])

                        # Create a vector full of ones for efficiency shocks
                        a_efficiency = pd.Series(index=betai_hat['sector'].index, data=1).to_frame()  # efficiency vector
                        a_efficiency = a_efficiency.rename(columns={a_efficiency.columns[0]: domestic_country})
                        a_efficiency = a_efficiency.reindex(psi.columns, axis=1, fill_value=1.0)
                        a_efficiency.index.names = ['Sector']

                        if config['efficiency']['activated']:
                            a_efficiency.loc[ENERGY_SECTORS, domestic_country] = config['efficiency']['value']  # to match the same IO effect as in the other shock scenarios

                        reference_parameters = deepcopy(config['reference_parameters'])
                        reference_parameters[param] = elasticity

                        if config['new_consumer']['activated']:
                            for share in config['new_consumer']['share']:
                                logging.info(f"Share of new consumer {share}")
                                phi = add_final_consumer_share(calib.phi, domestic_country, share_new_consumer=share)
                                equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors,
                                                                     emissions, xsi, psi, phi, costs_energy_final,
                                                                     psi_energy, psi_non_energy, costs_durable_final,
                                                                     psi_durable, psi_non_durable,
                                                                     costs_energy_services_final, Omega, costs_energy,
                                                                     Omega_energy, Omega_non_energy, Gamma, Leontieff,
                                                                     Domestic, Delta, sectors_dirty_energy,
                                                                     final_use_dirty_energy,
                                                                     share_GNE, domestic_country, descriptions,
                                                                     reference_parameters, config['new_consumer']['activated'],share)

                                save_path = get_save_path(reference_parameters, domestic_country, s, config['uniform_shock'], new_consumer=config['new_consumer']['activated'], share=share)
                                equilibrium_output.to_excel(Path(folder_to_save) / Path(save_path))
                        else:

                            equilibrium_output = run_equilibrium(li_hat, ki_hat, betai_hat, a_efficiency, sectors, emissions,
                                                                 xsi, psi,
                                                                 phi, costs_energy_final,
                                                                 psi_energy, psi_non_energy, costs_durable_final, psi_durable,
                                                                 psi_non_durable,
                                                                 costs_energy_services_final, Omega, costs_energy, Omega_energy,
                                                                 Omega_non_energy, Gamma, Leontieff,
                                                                 Domestic, Delta, sectors_dirty_energy, final_use_dirty_energy,
                                                                 share_GNE, domestic_country, descriptions,
                                                                 reference_parameters, config['new_consumer']['activated'],
                                                                 config['new_consumer']['share'])

                            save_path = get_save_path(reference_parameters, domestic_country, s, config['uniform_shock'], new_consumer=config['new_consumer']['activated'])
                            equilibrium_output.to_excel(Path(folder_to_save) / Path(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sufficiency production network simulations.")
    parser.add_argument('--config', type=str, required=True, default='config_reference.json', help="Path to the json configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_simulation(config)
