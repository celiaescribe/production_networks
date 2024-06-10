import numpy as np
import pandas as pd
from typing import Sequence
import json
from pathlib import Path

from dataclasses import dataclass
import os

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

@dataclass
class EquilibriumOutput:
    """Class to save the outcome of the model."""
    pi_hat: pd.DataFrame
    yi_hat: pd.DataFrame
    price_production: pd.DataFrame
    pi_imports_finaldemand: pd.DataFrame
    final_demand: pd.DataFrame
    intermediate_demand: pd.DataFrame
    domar: pd.DataFrame
    labor_capital: pd.DataFrame
    emissions_hat: pd.DataFrame
    emissions_detail: pd.DataFrame
    global_variables: pd.Series
    descriptions: pd.Series

    def to_excel(self, path):
        with pd.ExcelWriter(path) as writer:
            for current_df, sheet_name in [
                (self.pi_hat, "pi_hat"),
                (self.yi_hat, "yi_hat"),
                (self.price_production, "price_production"),
                (self.pi_imports_finaldemand, "pi_imports_finaldemand"),
                (self.final_demand, "final_demand"),
                (self.intermediate_demand, "intermediate_demand"),
                (self.domar, "domar"),
                (self.labor_capital, "labor_capital"),
                (self.emissions_hat, "emissions_hat"),
                (self.emissions_detail, "emissions_detail"),
                (self.global_variables, "global_variables"),
                (self.descriptions, "descriptions"),
            ]:
                is_dataframe = type(current_df) is not pd.Series

                # Copy the dataframe to avoid modifying the original one
                df_to_write = current_df.copy()
                # Add long description if the index has a "Sector" level
                if "Sector" in current_df.index.names and sheet_name != "descriptions" and sheet_name != "emissions_detail" and sheet_name != "intermediate_demand":
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
            price_production = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="price_production", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            pi_imports_finaldemand = pd.read_excel(xls, sheet_name="pi_imports_finaldemand", index_col=0).drop(columns="long_description")
            final_demand = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="final_demand", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            intermediate_demand = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="intermediate_demand", index_col=0),
                axis=0, level_names=["Country", "Sector"])
            domar = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="domar", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"]
            )
            labor_capital = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="labor_capital", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"]
            )
            emissions_hat = pd.read_excel(xls, sheet_name="emissions_hat", index_col=0)
            emissions_detail = pd.read_excel(xls, sheet_name="emissions_detail", index_col=0)
            global_variables = pd.read_excel(xls, sheet_name="global_variables", index_col=0, header=None).squeeze()
            descriptions = pd.read_excel(xls, sheet_name="descriptions", index_col=0, header=None).squeeze()
            descriptions.index.name = "Sector"
            descriptions.name = 0
        return cls(pi_hat, yi_hat, price_production, pi_imports_finaldemand, final_demand, intermediate_demand, domar, labor_capital, emissions_hat, emissions_detail, global_variables, descriptions)

def add_long_description(df, descriptions):
    "Add the long description of the sectors to the DataFrame"
    out = (
        df
        # Remove the multiindex
        .reset_index()
        # Merge on the column sector
        .merge(descriptions.reset_index(), on="Sector")
        # Rename the newly added column named "0"
        .rename(columns={0: "long_description"})
        # Put back the right index
        .set_index(df.index.names)
    )
    # Put the long description as the first column
    out = out[['long_description'] + out.columns[:-1].tolist()]
    return out


def flatten_index(index):
    "If a MultiIndex, flattens it, and name the new index with the names of the original index joined by '-'."
    if type(index) is not pd.MultiIndex:
        return index

    out = index.map('-'.join)
    out.name = '-'.join(index.names)
    return out

def unflatten_index(flattened_index, level_names: Sequence[str]):
    out = pd.Index([tuple(item.split('-')) for item in flattened_index])
    out.names = level_names  # type: ignore
    return out

def unflatten_index_in_df(df, axis: list[int] | int, level_names):
    if type(axis) is int:
        axis = [axis]
    if 0 in axis:  # type: ignore
        df.index = unflatten_index(df.index, level_names)
    if 1 in axis:  # type: ignore
        df.columns = unflatten_index(df.columns, level_names)
    return df


def same_df(df1, df2):
    """Check whether two dataframes are the same."""
    return (
            df1.index.equals(df2.index)
            and df1.columns.equals(df2.columns)
            and df1.dtypes.equals(df2.dtypes)
            and np.allclose(df1.values, df2.values)
    )


def load_config(config_file):
    """Load configuration file."""
    with open(Path('inputs/configs') / Path(config_file), 'r') as f:
        return json.load(f)


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


def get_save_path(reference_parameters, country, s, uniform, new_consumer, labor=False, share=0):
    """Create the path to save the results of the model.
    Args:
        reference_parameters: dict, parameters of the model
        country: str, name of the country
        s: str, name of the scenario
        uniform: bool, whether the uniform shock is used
        efficiency: bool, whether efficiency shock is used
        new_consumer: bool, whether a new consumer is used
        share: float, share of the new consumer if new_consumer is activated"""
    theta, sigma, epsilon, delta, mu, nu, kappa, rho = reference_parameters['theta'], reference_parameters['sigma'], reference_parameters['epsilon'], reference_parameters['delta'], reference_parameters['mu'], reference_parameters['nu'], reference_parameters['kappa'], reference_parameters['rho']
    if uniform:
        u = '_uniform'
    else:
        u = ''
    if new_consumer:
        c = '_heterogeneous'
        share = str(share)
    else:
        c = ''
        share = ''
    if labor:
        l = '_labor'
    else:
        l = ''
    path = f"{country}_{s}_theta{theta}_sigma{sigma}_epsilon{epsilon}_delta{delta}_mu{mu}_nu{nu}_kappa{kappa}_rho{rho}{u}{c}{share}{l}.xlsx"
    return path

