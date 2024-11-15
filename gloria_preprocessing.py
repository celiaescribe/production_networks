# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Original author Célia Escribe <celia.escribe@gmail.com>

import pandas as pd
import country_converter as coco
# import pycountry
import os
import pickle
from numpy.linalg import inv
import numpy as np
import re
import gc
from pathlib import Path
import argparse

"""Preprocessing module of the model
This module extracts aggregated inputs from the GLORIA database. In particular, it extracts input-output, final demand, value-added and emissions tables for a specified aggregation.
"""

data_spec_year = {
    2014: {
        'MRIO': '20230314',
        'satellite': '20230727'
    },
    2018: {
        'MRIO':'20230315',
        'satellite': '20230310'
    }
}

def read_input_data(year):
    """Read input data from GLORIA database, and specifies the year."""
    spec_mrio = data_spec_year[year]['MRIO']
    spec_satellite = data_spec_year[year]['satellite']
    T = pd.read_csv(
        f"GLORIA_MRIOs_57_{year}/{spec_mrio}_120secMother_AllCountries_002_T-Results_{year}_057_Markup001(full).csv",
        header=None)  # transaction matrix
    V = pd.read_csv(
        f"GLORIA_MRIOs_57_{year}/{spec_mrio}_120secMother_AllCountries_002_V-Results_{year}_057_Markup001(full).csv",
        header=None)  # value added matrix
    Y = pd.read_csv(
        f"GLORIA_MRIOs_57_{year}/{spec_mrio}_120secMother_AllCountries_002_Y-Results_{year}_057_Markup001(full).csv",
        header=None)  # final demand matrix

    TQ = pd.read_csv(
        f"GLORIA_MRIOs_57_{year}/{spec_satellite}_120secMother_AllCountries_002_TQ-Results_{year}_057_Markup001(full).csv",
        header=None)  # satellite accounts, intermediate matrix
    YQ = pd.read_csv(
        f"GLORIA_MRIOs_57_{year}/{spec_satellite}_120secMother_AllCountries_002_YQ-Results_{year}_057_Markup001(full).csv",
        header=None)  # satellite accounts, final demand
    # commodity_prices = pd.read_csv(
    #     "GLORIA_MRIOs_57_2014/GLORIA_MRIO_Loop059_part_IV_commodityprices/20240111_120secMother_AllCountries_002_Prices_2014_059_Markup001(full).csv",
    #     header=None)

    # Read Excel sheets
    region = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Regions")
    sectors = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sectors")
    labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sequential region-sector labels")
    sat_labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Satellites")

    # Create abbreviation of sector names (example for a few sectors)
    sectors['sec_short'] = ["Wheat", "Maize", "CerOth", "Legume", "Rice", "Veget", "Sugar", "Tobac", "Fibre", "CropOth",
                            "Grape", "FruitNut", "CropBev", "SpicePhar", "Seed",
                            "Cattle", "Sheep", "Pig", "Poultry", "AnimOth", "Forest", "Fish", "Crust", "Coal",
                            "Lignite",
                            "Petrol", "Gas", "IronOres", "UranOres", "AluOres",
                            "CopperOres", "GoldOres", "LedZinSilOres", "NickelOres", "TinOres", "NonferOthOres",
                            "StoneSand", "ChemFert", "Salt", "OthServ", "BeefMeat",
                            "SheepMeat", "PorkMeat", "PoultryMeat", "MeatOth", "FishProd", "Cereal", "VegetProd",
                            "Fruit",
                            "FoodOth", "Sweets", "FatAnim", "FatVeget", "Dairy",
                            "Beverage", "TobacProd", "Textile", "Leather", "Sawmill", "Paper", "Print", "Coke", "Petro",
                            "NFert", "OthFert", "ChemPetro", "ChemInorg", "ChemOrg",
                            "Pharma", "ChemOth", "Rubber", "Plastic", "Clay", "Ceramics", "Cement", "MinOth",
                            "IronSteel",
                            "Alu", "Copper", "Gold", "LedZinSil", "Nickel", "Tin",
                            "NonferOth", "FabMetal", "Machine", "Vehicle", "TransOth", "Repair", "Electronic",
                            "Electrical",
                            "FurnOth", "Power", "FuelDist", "Water", "Waste",
                            "Recovery", "ConstBuild", "ConstCivil", "TradeRep", "Road", "Rail", "Pipe", "WaterTrans",
                            "Air",
                            "Services", "Post", "Hospitality", "Publishing",
                            "Telecom", "Info", "Finance", "Real", "ProfSci", "Admin", "Public", "Edu", "Health", "Arts",
                            "Oth"]

    # Rename column names in labels DataFrame
    labels.columns = ["Lfd_Nr", "io_lab", "fd_lab", "va_lab"]
    return T, V, Y, TQ, YQ, region, sectors, labels, sat_labels

#

# commodity_prices.columns = sectors['Sector_names']

def define_region_mapping(region, country):
    """Function to map the country names to aggregated regions."""
    # Convert ISO3 to UN region names and handle exceptions
    region['un_cat'] = region['Region_acronyms'].apply(lambda x: coco.convert(names=x, to='continent') if x not in ['XAF', 'XAM', 'XAS', 'XEU', 'SDS', 'DYE'] else None)

    # # Assuming region is a DataFrame with 'Region_acronyms' column
    # region['un_cat'] = region['Region_acronyms'].apply(lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None)
    #
    region.loc[region['Region_acronyms'] == 'XAF', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'XAM', 'un_cat'] = 'America'
    region.loc[region['Region_acronyms'] == 'XAS', 'un_cat'] = 'Asia'
    region.loc[region['Region_acronyms'] == 'XEU', 'un_cat'] = 'Europe'
    region.loc[region['Region_acronyms'] == 'SDS', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'DYE', 'un_cat'] = 'Asia'

    region_mapping = dict(zip(region['Region_names'], region['un_cat']))

    # get unique values from region_mapping
    unique_values = set(region_mapping.values())

    if country == 'EU':
        row_mapping = {
            'Africa': 'RoW',
            'Americas': 'RoW',
            'Asia': 'RoW',
            'Europe': 'RoW',  # Default to RoW for Europe, will handle EU separately
            'Oceania': 'RoW',
            'America': 'RoW'
        }

        eu_countries = [
            'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU',
            'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT',
            'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
        ]

        region_mapping = {c: (
            'EU' if region['Region_acronyms'][region['Region_names'] == c].values[0] in eu_countries else 'ROW') for c in region_mapping.keys()}

    elif country == 'Europe':
        row_mapping = {
            'Africa': 'RoW',
            'Americas': 'RoW',
            'Asia': 'RoW',
            'Europe': 'Europe',
            'Oceania': 'RoW',
            'America': 'RoW'
        }

        region_mapping = {c: row_mapping[region_mapping[c]] for c in region_mapping.keys()}
    else:
        region_mapping[country] = country  # country is kept separatly

        row_mapping = {
            country: country,
            'Africa': 'RoW',
            'Americas': 'RoW',
            'Asia': 'RoW',
            'Europe': 'RoW',
            'Oceania': 'RoW',
            'America': 'RoW'
        }

        region_mapping = {c: row_mapping[region_mapping[c]] for c in region_mapping.keys()}
    return region_mapping

def map_country_to_region(country, region_mapping):
    return region_mapping.get(country, country)


def preprocess_io_data(T, Y, V, labels, region_mapping):
    """Function that takes as inputs the raw IO data (supply-use and final demand) and that creates aggregated tables that can be used for calibration.
    T: pd.DataFrame
        Intermediary consumption matrix
    Y: pd.DataFrame
        Final demand matrix
    V: pd.DataFrame
        Value added matrix
    labels: pd.DataFrame
        Sequential region-sector labels
    region_mapping: dict
        Mapping of countries to regions
    """
    # Adding original labels to the inputs
    T.columns = labels['io_lab']
    T.index = labels['io_lab']
    Y.columns = labels['fd_lab'].dropna()
    Y.index = labels['io_lab']
    V.columns = labels['io_lab']
    V.index = labels['va_lab'].dropna()
    # #
    # Create vectors for industry and product labels
    industry_labs = labels[labels['io_lab'].str.contains("industry")]['io_lab']
    product_labs = labels[~labels['io_lab'].isin(industry_labs)]['io_lab']
    # #
    # Get rid of some countries with problematic names
    list_countries_drop = ['Yemen Arab Republic/Yemen', 'Yugoslavia/Serbia', 'Zambia', 'Zimbabwe', 'CSSR/Czech Republic',
                           'Ethiopia/DR Ethiopia', 'USSR/Russian Federation', 'Sudan/North Sudan', 'DR Yemen']
    list_countries_drop = ['Yemen Arab Republic/Yemen (1990/1991)', 'DR Yemen (Aden)']
    # #
    # # Extracting the use matrix, final demand matrix, value added matrix, and satellite accounts
    Z = T.loc[product_labs, industry_labs]  # use matrix
    Y = Y.loc[product_labs]  # final demand matrix
    V = V[industry_labs]  # value added matrix

    V.index = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(i) for i in V.index], names=['Country', 'Detail'])
    V.columns = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(c) for c in V.columns], names=['Country', 'Industry'])
    rows_to_drop = V.index.get_level_values('Country').isin(list_countries_drop)
    rows_to_drop = V.index[rows_to_drop]

    # Filter the MultiIndex for columns
    cols_to_drop = V.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = V.columns[cols_to_drop]

    # Drop the specified rows and columns
    V = V.drop(index=rows_to_drop, columns=cols_to_drop)
    V.index = V.index.remove_unused_levels()
    V.columns = V.columns.remove_unused_levels()

    # Apply the mapping to the index and columns
    V.index = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), detail) for country, detail in V.index])
    V.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), industry) for country, industry in V.columns])

    # Group by regions, and sum the values
    V = V.groupby(level=[0, 1], axis=0).sum()
    V = V.groupby(level=[0, 1], axis=1).sum()

    V.index = V.index.droplevel(0)  # we drop the country level
    V = V.groupby(level=0, axis=0).sum()  # we sum all the rows by level of value-added, as this is a block-diagonal matrix

    # V.index = [' - '.join(idx) for idx in V.index]
    V.columns = [' - '.join(col) for col in V.columns]
    V.columns = [c.replace(' industry', '') for c in V.columns]

    Y.index = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(i) for i in Y.index], names=['Country', 'Industry'])
    Y.columns = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(c) for c in Y.columns], names=['Country', 'Detail'])

    rows_to_drop = Y.index.get_level_values('Country').isin(list_countries_drop)
    rows_to_drop = Y.index[rows_to_drop]
    cols_to_drop = Y.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = Y.columns[cols_to_drop]

    Y = Y.drop(index=rows_to_drop, columns=cols_to_drop)
    Y.index = Y.index.remove_unused_levels()
    Y.columns = Y.columns.remove_unused_levels()

    # Apply the mapping to the index and columns
    Y.index = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), detail) for country, detail in Y.index])  # rename countries
    Y.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), industry) for country, industry in Y.columns])  # rename countries
    Y = Y.groupby(level=[0, 1], axis=0).sum()  # sum for similar countries
    Y = Y.groupby(level=[0, 1], axis=1).sum()
    Y.index = [' - '.join(idx) for idx in Y.index]  # rename index
    Y.columns = [' - '.join(col) for col in Y.columns]
    Y.index = [replace_last_occurrence(i, ' product', '') for i in Y.index]

    Z.index = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(i) for i in Z.index], names=['Country', 'Industry'])
    Z.columns = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(c) for c in Z.columns], names=['Country', 'Industry'])

    rows_to_drop = Z.index.get_level_values('Country').isin(list_countries_drop)
    rows_to_drop = Z.index[rows_to_drop]
    cols_to_drop = Z.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = Z.columns[cols_to_drop]
    Z = Z.drop(index=rows_to_drop, columns=cols_to_drop)
    Z.index = Z.index.remove_unused_levels()
    Z.columns = Z.columns.remove_unused_levels()

    Z.index = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), detail) for country, detail in Z.index])
    Z.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), industry) for country, industry in Z.columns])
    Z = Z.groupby(level=[0, 1], axis=0).sum()
    Z = Z.groupby(level=[0, 1], axis=1).sum()
    Z.index = [' - '.join(idx) for idx in Z.index]
    Z.columns = [' - '.join(col) for col in Z.columns]
    Z.index = [replace_last_occurrence(i, ' product', '') for i in Z.index]
    Z.columns = [c.replace(' industry', '') for c in Z.columns]
    return Z, Y, V

def preprocess_emissions_data(TQ, YQ, labels, sat_labels, system='OECD'):
    """Function that takes as inputs the raw emissions data (supply and final demand) and that creates aggregated tables that can be used for calibration.
    TQ: pd.DataFrame
        Satellite data for each productive sector
    YQ: pd.DataFrame
        Satellite data for final demand
    labels: pd.DataFrame
        Sequential region-sector labels
    sat_labels: pd.DataFrame
        Satellite labels
    system: str
        System for which the emissions are extracted (e.g., 'OECD')
    """
    industry_labs = labels[labels['io_lab'].str.contains("industry")]['io_lab']
    product_labs = labels[~labels['io_lab'].isin(industry_labs)]['io_lab']

    # Get rid of some countries with problematic names
    list_countries_drop = ['Yemen Arab Republic/Yemen', 'Yugoslavia/Serbia', 'Zambia', 'Zimbabwe', 'CSSR/Czech Republic',
                           'Ethiopia/DR Ethiopia', 'USSR/Russian Federation', 'Sudan/North Sudan', 'DR Yemen']
    list_countries_drop = ['Yemen Arab Republic/Yemen (1990/1991)', 'DR Yemen (Aden)']
    #
    # Adding labels to inputs
    TQ.columns = labels['io_lab']
    TQ.index = sat_labels['Sat_indicator'] + " - " + sat_labels['Sat_unit']
    #
    YQ.columns = labels['fd_lab'].dropna()
    YQ.index = sat_labels['Sat_indicator'] + " - " + sat_labels['Sat_unit']

    YQ.columns = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(c) for c in YQ.columns], names=['Country', 'Detail'])
    cols_to_drop = YQ.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = YQ.columns[cols_to_drop]
    YQ = YQ.drop(columns=cols_to_drop)
    YQ.columns = YQ.columns.remove_unused_levels()
    YQ.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), detail) for country, detail in YQ.columns])
    YQ = YQ.groupby(level=[0, 1], axis=1).sum()
    #
    ZQ = TQ[industry_labs]  # satellite accounts
    ZQ.columns = pd.MultiIndex.from_tuples([split_string_on_final_parenthesis(c) for c in ZQ.columns], names=['Country', 'Industry'])
    cols_to_drop = ZQ.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = ZQ.columns[cols_to_drop]
    ZQ = ZQ.drop(columns=cols_to_drop)
    ZQ.columns = ZQ.columns.remove_unused_levels()
    ZQ.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country, region_mapping), industry) for country, industry in ZQ.columns])
    ZQ = ZQ.groupby(level=[0, 1], axis=1).sum()
    ZQ.columns = [' - '.join(col) for col in ZQ.columns]
    ZQ.columns = [c.replace(' industry', '') for c in ZQ.columns]

    # We  extract CO2 emissions from intermediate consumptions
    # We take total values for each of the gases, using the OECD classification
    emissions_total_Z = ZQ.loc[ZQ.index.str.contains('total'),:]
    emissions_total_Z = emissions_total_Z.loc[emissions_total_Z.index.str.contains(system), :]
    emissions_total_Z = emissions_total_Z.T
    emissions_total_Z = emissions_total_Z.loc[:, ~emissions_total_Z.columns.duplicated()]
    new_order = [f"'GHG_total_{system}_consistent' - kilotonnes CO2-equivalent", f"'co2_excl_short_cycle_org_c_total_{system}_consistent' - kilotonnes", f"'ch4_total_{system}_consistent' - kilotonnes", f"'n2o_total_{system}_consistent' - kilotonnes"]
    rest = [c for c in emissions_total_Z.columns if c not in new_order]
    new_order = new_order + rest
    emissions_total_Z = emissions_total_Z.reindex(columns = new_order)
    emissions_total_Z = emissions_total_Z.rename(columns={
        f"'GHG_total_{system}_consistent' - kilotonnes CO2-equivalent": f"GHG_total_{system}_consistent_ktco2eq"
    })
    emissions_total_Z.columns = [emissions_total_Z.columns[0]] + ['_'.join(''.join(idx.split("'")[1:]).split(' - ')) for idx in emissions_total_Z.columns[1:]]

    # We now focus on detailed emissions, to define the share of emissions associated to energy-related emissions and the share of emissions associated to process-related emissions
    emissions_detail = ZQ.loc[ZQ.index.str.contains(system),:]
    emissions_detail = emissions_detail.T
    emissions_detail = emissions_detail.loc[:, ~emissions_detail.columns.duplicated()]
    new_order = [f"'GHG_total_{system}_consistent' - kilotonnes CO2-equivalent", f"'co2_excl_short_cycle_org_c_total_{system}_consistent' - kilotonnes", f"'ch4_total_{system}_consistent' - kilotonnes", f"'n2o_total_{system}_consistent' - kilotonnes"]
    rest = [c for c in emissions_detail.columns if c not in new_order]
    new_order = new_order + rest
    emissions_detail = emissions_detail.reindex(columns = new_order)

    # We make the assumption that it is only the category 1 that includes energy-related emissions. We exclude categories 1.B from those, as we assume the corresponding fugitive emissions are
    # process-related emissions, and therefore not energy-related emissions
    list_emissions_energy = ['1A1a', '1A1bc', '1A1b', '1A1ci', '1A1cii', '1A2', '1A2a', '1A2b', '1A2c', '1A2d', '1A3e', '1A2f',
                             '1A2g', '1A2h', '1A2i', '1A2j', '1A2k', '1A2l', '1A2m', '1A3b', '1A3b_noRES', '1A3b_RES', '1A3a', '1A3c', '1A3d', '1A3e', '1A4', '1A4a', '1A4b', '1A4ci',
                             '1A4cii', '1A4ciii', '1A5']  # activities related to energy GHG emissions
    list_gas = ['co2']

    def create_new_index(names):
        """Groupby type of gas (CO2, CH4, etc...) and type of activity (1A1a, 1A1b, etc...)"""
        new_index = []
        for name in names:
            parts = name.split('_')
            # Check for special cases 'noRES' or 'RES'
            if 'noRES' in parts or 'RES' in parts:
                # Include the qualifier (e.g., 'noRES' or 'RES') in the second level
                first_level = '_'.join(parts[:-3][:-2])  # Exclude the last 4 parts for the first level
                second_level = '_'.join(parts[:-3][-2:])  # Include 'noRES' or 'RES' in the second level
            else:
                # General case handling
                first_level = '_'.join(parts[:-3][:-1])  # Exclude the last 3 parts for the first level
                second_level = parts[:-3][-1]  # The category identifier is the third-last part
            new_index.append((first_level, second_level))
        return pd.MultiIndex.from_tuples(new_index)

    emissions_energy = emissions_detail[emissions_detail.columns[~emissions_detail.columns.str.contains('total')]]
    # We calculate approximately the share of emissions associated to CO2 (excl_short_cycle, not org_short_cycle) or CH4 for total emissions for each sector
    emissions_total = emissions_detail[emissions_detail.columns[emissions_detail.columns.str.contains('total')]]
    emissions_total = emissions_total[
        emissions_total.columns[emissions_total.columns.str.contains('|'.join(['GHG', 'co2', 'ch4']))]]
    emissions_total = emissions_total.div(emissions_total[f"'GHG_total_{system}_consistent' - kilotonnes CO2-equivalent"], axis=0)
    emissions_total["co2_excl_short_cycle_org_c"] = emissions_total[f"'co2_excl_short_cycle_org_c_total_{system}_consistent' - kilotonnes"]
    emissions_total["ch4"] = 1 - emissions_total[f"'co2_excl_short_cycle_org_c_total_{system}_consistent' - kilotonnes"]  # we make the assumption that the rest of emissions is CH4 in proportion, which is a big assumption
    emissions_total = emissions_total[["co2_excl_short_cycle_org_c", "ch4"]]

    # TODO: j'observe qqc d'un peu surprenant qui est que pour plusieurs secteurs, co2_excl_short_cycle_org_c est pas mal plus faible que GHG_total. Je ne suis pas certaine de ce que cela implique pour les datas.
    # TODO: cette approximation pose en partie problème pour les secteurs agricoles, où les émissions ne vont pas forcément être du méthane, mais peuvent être aussi du N2O, qui n'est donc pas capté ici.
    # TODO: Mais cependant, cela n'a pas l'air de poser trop de problème pour mes calculs et estimation finale du rôle des émissions process-related qui sont bien cohérentes.

    # Remark: we choose to focus on CO2 and CH4 for two reasons. The first is that since we need the activity classification inputs, and we only have that for each gas (and not for total GHG-eq), we need to focus on specific gases.
    # Second, estimating the share of emissions linked to a specific gas requires to be able to convert CO2-eq emissions into CO2 emissions. This is difficult. By just assuming that emissions are either CO2 or CH4, we avoid this problem.
    # CO2 emissions are directly in the good unit, and the rest is assumed to be all methane. Of course, this may lead to a somehow biased estimate for the origin of the emissions (process- or energ-related).

    # emissions_energy = emissions_detail[emissions_detail.columns[emissions_detail.columns.str.contains('|'.join(list_emissions_energy))]]
    emissions_energy.columns = ['_'.join(''.join(idx.split("'")[1:]).split(' - ')) for idx in emissions_energy.columns]  # rename columns

    # new_index = pd.MultiIndex.from_tuples([('_'.join(idx.split('_')[:-3][:-1]), idx.split('_')[:-3][-1]) for idx in emissions_energy.columns])
    new_index = create_new_index(emissions_energy.columns)
    emissions_energy.columns = new_index  # groupby type of gas and type of activity

    mask = emissions_energy.columns.get_level_values(1).isin(list_emissions_energy)
    filtered_columns = emissions_energy.columns[mask]

    # We calculate for each gas the share of emissions coming from direct energy emissions
    grouped = emissions_energy.groupby(axis=1, level=0)
    energy_related_emissions = {}
    for name, group in grouped:
        # Sum of columns that match list_emissions_energy within the group
        sum_match = group[filtered_columns.intersection(group.columns)].sum(axis=1)
        # Sum of all columns within the group
        sum_all = group.sum(axis=1)
        # Calculate the ratio
        ratio = sum_match / sum_all
        energy_related_emissions[name] = ratio

    energy_related_emissions = pd.DataFrame(energy_related_emissions)
    energy_related_emissions = energy_related_emissions.fillna(0)  # for each gas (in columns), we have the share of energy-related emissions based on activity classification
    # We calculate approximately overall the share of GHG emissions which can be associated to energy-related emissions by using only co2 and ch4 as explained before
    energy_related_emissions = energy_related_emissions[['co2_excl_short_cycle_org_c', 'ch4']]
    energy_related_emissions = (energy_related_emissions * emissions_total).sum(axis=1)  # based on our approximation focusing only on co2 and ch4, we calculate the share of energy-related emissions for each sector

    emissions_total_Z = pd.concat([emissions_total_Z, energy_related_emissions.rename("share_energy_related")], axis=1)

    # Approximation: final demand emissions is just CO2 emissions
    emissions_total_y = YQ.loc[YQ.index.str.contains('co2_excl_short_cycle_org'), :]
    emissions_total_y = emissions_total_y.loc[emissions_total_y.index.str.contains(system), :]
    # we allocate emissions of final consumption to a given sector. This is a bit arbitrary probably, notably for the distribution of gaseous fuels through mains
    # We assume that transport emissions are only petroleum, while the rest (housing ?) is gaseous fuels (while it could be petroleum)
    emissions_total_y = emissions_total_y.rename(index={
        f"'co2_excl_short_cycle_org_c_1A3b_noRES_{system}_consistent' - kilotonnes": 'Refined petroleum products',
        f"'co2_excl_short_cycle_org_c_1A4_{system}_consistent' - kilotonnes": 'Distribution of gaseous fuels through mains'  # this is a strong hypothesis, as many emissions from heating may come from other energy sources
    })
    emissions_total_y = emissions_total_y.loc[['Refined petroleum products', 'Distribution of gaseous fuels through mains'],:]
    emissions_total_y = emissions_total_y.groupby(level=0, axis=1).sum()
    sectors_to_add = pd.Index([e.split(' - ')[1] for e in emissions_total_Z.index])
    sectors_to_add = sectors_to_add.difference(emissions_total_y.index)
    additional_rows = pd.DataFrame(0, index=sectors_to_add, columns=emissions_total_y.columns)
    emissions_total_y = pd.concat([emissions_total_y, additional_rows], axis=0)  # we add other rows to match the shape of emissions_total_Z, which will correspond to zero emissions
    emissions_total_y = emissions_total_y.stack().reset_index()
    emissions_total_y.columns = ['sector', 'country', f"GHG_total_{system}_consistent_ktco2eq"]
    emissions_total_y['index'] = emissions_total_y['country'] + ' - ' + emissions_total_y['sector']
    emissions_total_y = emissions_total_y.set_index('index')[f"GHG_total_{system}_consistent_ktco2eq"].to_frame()  # be careful, here the country corresponds to the consumption of the final sector

    emissions_total = pd.concat([emissions_total_Z, emissions_total_y], axis=0)
    return emissions_total, emissions_total_Z, emissions_total_y

def split_string(s):
    """Function to split the string and exclude the parentheses content"""
    parts = re.match(r'(.*) \([^)]*\) (.*)', s)
    if parts:
        return parts.group(1), parts.group(2)
    else:
        return s, ''

def split_string_new(s):
    # Use a regular expression to split the string at the first parentheses, considering nested parentheses
    parts = re.match(r'^(.*?)\(([^)]*)\)\s*(.*)$', s)
    if parts:
        country = parts.group(1).strip()
        # Additional content inside the first parentheses (e.g., country code) is captured but not used
        detail = parts.group(3).strip()
        return country, detail
    else:
        return s, ''


def split_string_on_final_parenthesis(s):
    """Splits country names with the rest (this can be a product, value chain or final demand specification"""
    special_case_1 = "Growing beverage crops (coffee, tea etc) industry"
    if special_case_1 in s:  # special case handling:
        part1, part2 = s.split(special_case_1)[0], 'Growing beverage crops (coffee, tea etc) industry'
        if part1.strip().endswith(')'):
            part1 = part1.strip()[:part1.rfind('(')].strip()
        return part1, part2

    special_case_2 = "Growing beverage crops (coffee, tea etc) product"
    if special_case_2 in s:  # special case handling:
        part1, part2 = s.split(special_case_2)[0], 'Growing beverage crops (coffee, tea etc) product'
        if part1.strip().endswith(')'):
            part1 = part1.strip()[:part1.rfind('(')].strip()
        return part1, part2

    # Find the index of the final closing parenthesis
    index = s.rfind(')')
    # If the closing parenthesis is found, split the string
    if index != -1:
        part1 = s[:index + 1].strip()
        part2 = s[index + 1:].strip()
        # Remove the final parenthesis from part1 if it exists
        if part1.endswith(')'):
            part1 = part1[:part1.rfind('(')].strip()
        return part1, part2
    else:
        return s, ''

def replace_last_occurrence(s, target, replacement):
    """Replaces the last occurence of target in string s. We uses complex reversing techniques, because
    there is easy solution to do so in python"""
    return s[::-1].replace(target[::-1], replacement[::-1], 1)[::-1]


def aggregate_data(data, country_groups):
    # Split column and index to MultiIndex
    data.columns = pd.MultiIndex.from_tuples([split_string(c) for c in data.columns], names=['Country', 'Industry'])
    data.index = pd.MultiIndex.from_tuples([split_string(i) for i in data.index], names=['Country', 'Detail'])

    # Aggregate countries according to the country_groups dictionary
    new_data = pd.DataFrame(index=data.index, columns=pd.MultiIndex.from_product([country_groups.keys(), data.columns.get_level_values(1).unique()]))

    for group, countries in country_groups.items():
        group_data = data.loc[:, data.columns.get_level_values('Country').isin(countries)]
        group_sum = group_data.groupby(level='Industry', axis=1).sum(min_count=1)
        new_data[group] = group_sum

    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GLORIA database.')
    parser.add_argument("--country", type=str, default='France', help="Country to do the processing")
    parser.add_argument("--year", type=int, default=2014, help="Year to use in the IO table")
    args = parser.parse_args()
    country = args.country  # we select the config we are interested in
    year = int(args.year)

    T, V, Y, TQ, YQ, region, sectors, labels, sat_labels = read_input_data(year)

    region_mapping = define_region_mapping(region, country)
    Z, Y, V = preprocess_io_data(T, Y, V, labels, region_mapping)
    emissions_total, emissions_total_Z, emissions_total_y = preprocess_emissions_data(TQ, YQ, labels, sat_labels, system='OECD')

    data_path = f"GLORIA_MRIOs_57_{year}/"

    country = country.lower()
    if len(country.split(' ')) > 1:
        country = '_'.join(country.split(' '))  # we process the name of the country for saving it
    #
    Z.to_pickle(Path(data_path) / Path(f"Z_{country}_RoW_{year}.pkl"))
    Y.to_pickle(Path(data_path) / Path(f"Y_{country}_RoW_{year}.pkl"))
    V.to_pickle(Path(data_path) / Path(f"V_{country}_RoW_{year}.pkl"))
    emissions_total_Z.to_pickle(Path(data_path) / Path(f"emissions_Z_{country}_RoW_{year}.pkl"))
    emissions_total_y.to_pickle(Path(data_path) / Path(f"emissions_Y_{country}_RoW_{year}.pkl"))
    #
    # Save other objects
    with open(Path(data_path) / Path("sectors.pkl"), 'wb') as f:
        pickle.dump(sectors, f)
