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

# Function to split the string and exclude the parentheses content
def split_string(s):
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
    args = parser.parse_args()
    country = args.country  # we select the config we are interested in

    # # Read CSV files
    T = pd.read_csv("GLORIA_MRIOs_57_2014/20230314_120secMother_AllCountries_002_T-Results_2014_057_Markup001(full).csv", header=None)  # transaction matrix
    V = pd.read_csv("GLORIA_MRIOs_57_2014/20230314_120secMother_AllCountries_002_V-Results_2014_057_Markup001(full).csv", header=None)  # value added matrix
    Y = pd.read_csv("GLORIA_MRIOs_57_2014/20230314_120secMother_AllCountries_002_Y-Results_2014_057_Markup001(full).csv", header=None)  # final demand matrix

    #

    # Read Excel sheets
    region = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Regions")
    sectors = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sectors")
    labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sequential region-sector labels")
    sat_labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Satellites")
    #
    # Create abbreviation of sector names (example for a few sectors)
    sectors['sec_short'] = ["Wheat","Maize","CerOth","Legume","Rice","Veget","Sugar","Tobac","Fibre","CropOth","Grape","FruitNut","CropBev", "SpicePhar","Seed",
                            "Cattle","Sheep","Pig","Poultry","AnimOth","Forest","Fish","Crust","Coal","Lignite","Petrol", "Gas","IronOres","UranOres","AluOres",
                            "CopperOres","GoldOres","LedZinSilOres","NickelOres","TinOres","NonferOthOres", "StoneSand","ChemFert","Salt","OthServ","BeefMeat",
                            "SheepMeat","PorkMeat","PoultryMeat","MeatOth","FishProd","Cereal", "VegetProd","Fruit","FoodOth","Sweets","FatAnim","FatVeget","Dairy",
                            "Beverage","TobacProd","Textile","Leather","Sawmill", "Paper","Print","Coke","Petro","NFert","OthFert","ChemPetro","ChemInorg","ChemOrg",
                            "Pharma","ChemOth","Rubber","Plastic", "Clay","Ceramics","Cement","MinOth","IronSteel","Alu","Copper","Gold","LedZinSil","Nickel","Tin",
                            "NonferOth","FabMetal", "Machine","Vehicle","TransOth","Repair","Electronic","Electrical","FurnOth","Power","FuelDist","Water","Waste",
                            "Recovery", "ConstBuild","ConstCivil","TradeRep","Road","Rail","Pipe","WaterTrans","Air","Services","Post","Hospitality","Publishing",
                            "Telecom","Info","Finance","Real","ProfSci","Admin","Public","Edu","Health","Arts","Oth"]

    # Rename column names in labels DataFrame
    labels.columns = ["Lfd_Nr", "io_lab", "fd_lab", "va_lab"]
    #
    # Assuming 'region' is a DataFrame with a column 'Region_acronyms'

    # Convert ISO3 to UN region names and handle exceptions
    region['un_cat'] = region['Region_acronyms'].apply(lambda x: coco.convert(names=x, to='continent') if x not in ['XAF', 'XAM', 'XAS', 'XEU', 'SDS', 'DYE'] else None)

    # # Assuming region is a DataFrame with 'Region_acronyms' column
    # region['un_cat'] = region['Region_acronyms'].apply(lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None)
    #
    region.loc[region['Region_acronyms'] == 'XAF', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'XAM', 'un_cat'] = 'Americas'
    region.loc[region['Region_acronyms'] == 'XAS', 'un_cat'] = 'Asia'
    region.loc[region['Region_acronyms'] == 'XEU', 'un_cat'] = 'Europe'
    region.loc[region['Region_acronyms'] == 'SDS', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'DYE', 'un_cat'] = 'Asia'

    region_mapping = dict(zip(region['Region_names'], region['un_cat']))
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


    def map_country_to_region(country):
        return region_mapping.get(country, country)
    #
    #
    # Adding original labels to the data
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
    #
    # Get rid of some countries with problematic names
    list_countries_drop = ['Yemen Arab Republic/Yemen', 'Yugoslavia/Serbia', 'Zambia', 'Zimbabwe', 'CSSR/Czech Republic',
                           'Ethiopia/DR Ethiopia', 'USSR/Russian Federation', 'Sudan/North Sudan', 'DR Yemen']
    # #
    # # Extracting the use matrix, final demand matrix, value added matrix, and satellite accounts
    Z = T.loc[product_labs, industry_labs]  # use matrix
    Y = Y.loc[product_labs]  # final demand matrix
    V = V[industry_labs]  # value added matrix

    V.index = pd.MultiIndex.from_tuples([split_string_new(i) for i in V.index], names=['Country', 'Detail'])
    V.columns = pd.MultiIndex.from_tuples([split_string_new(c) for c in V.columns], names=['Country', 'Industry'])
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
    V.index = pd.MultiIndex.from_tuples([(map_country_to_region(country), detail) for country, detail in V.index])
    V.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country), industry) for country, industry in V.columns])

    # Group by the new index and columns, and sum the values
    V = V.groupby(level=[0, 1], axis=0).sum()
    V = V.groupby(level=[0, 1], axis=1).sum()

    V.index = V.index.droplevel(0)  # we drop the country level
    V = V.groupby(level=0, axis=0).sum()  # we sum all the rows by level of value-added, as this is a block-diagonal matrix

    # V.index = [' - '.join(idx) for idx in V.index]
    V.columns = [' - '.join(col) for col in V.columns]
    V.columns = [c.replace(' industry', '') for c in V.columns]

    Y.index = pd.MultiIndex.from_tuples([split_string_new(i) for i in Y.index], names=['Country', 'Industry'])
    Y.columns = pd.MultiIndex.from_tuples([split_string_new(c) for c in Y.columns], names=['Country', 'Detail'])

    rows_to_drop = Y.index.get_level_values('Country').isin(list_countries_drop)
    rows_to_drop = Y.index[rows_to_drop]
    cols_to_drop = Y.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = Y.columns[cols_to_drop]

    Y = Y.drop(index=rows_to_drop, columns=cols_to_drop)
    Y.index = Y.index.remove_unused_levels()
    Y.columns = Y.columns.remove_unused_levels()

    # Apply the mapping to the index and columns
    Y.index = pd.MultiIndex.from_tuples([(map_country_to_region(country), detail) for country, detail in Y.index])  # rename countries
    Y.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country), industry) for country, industry in Y.columns])  # rename countries
    Y = Y.groupby(level=[0, 1], axis=0).sum()  # sum for similar countries
    Y = Y.groupby(level=[0, 1], axis=1).sum()
    Y.index = [' - '.join(idx) for idx in Y.index]  # rename index
    Y.columns = [' - '.join(col) for col in Y.columns]
    Y.index = [replace_last_occurrence(i, ' product', '') for i in Y.index]

    Z.index = pd.MultiIndex.from_tuples([split_string_new(i) for i in Z.index], names=['Country', 'Industry'])
    Z.columns = pd.MultiIndex.from_tuples([split_string_new(c) for c in Z.columns], names=['Country', 'Industry'])

    rows_to_drop = Z.index.get_level_values('Country').isin(list_countries_drop)
    rows_to_drop = Z.index[rows_to_drop]
    cols_to_drop = Z.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = Z.columns[cols_to_drop]
    Z = Z.drop(index=rows_to_drop, columns=cols_to_drop)
    Z.index = Z.index.remove_unused_levels()
    Z.columns = Z.columns.remove_unused_levels()

    Z.index = pd.MultiIndex.from_tuples([(map_country_to_region(country), detail) for country, detail in Z.index])
    Z.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country), industry) for country, industry in Z.columns])
    Z = Z.groupby(level=[0, 1], axis=0).sum()
    Z = Z.groupby(level=[0, 1], axis=1).sum()
    Z.index = [' - '.join(idx) for idx in Z.index]
    Z.columns = [' - '.join(col) for col in Z.columns]
    Z.index = [replace_last_occurrence(i, ' product', '') for i in Z.index]
    Z.columns = [c.replace(' industry', '') for c in Z.columns]

    TQ = pd.read_csv("GLORIA_MRIOs_57_2014/20230727_120secMother_AllCountries_002_TQ-Results_2014_057_Markup001(full).csv", header=None)  # satellite accounts, intermediate matrix
    YQ = pd.read_csv("GLORIA_MRIOs_57_2014/20230727_120secMother_AllCountries_002_YQ-Results_2014_057_Markup001(full).csv", header=None)  # satellite accounts, final demand
    commodity_prices = pd.read_csv(
        "GLORIA_MRIOs_57_2014/GLORIA_MRIO_Loop059_part_IV_commodityprices/20240111_120secMother_AllCountries_002_Prices_2014_059_Markup001(full).csv", header=None)
    commodity_prices.columns = sectors['Sector_names']

    # Adding labels to data
    TQ.columns = labels['io_lab']
    TQ.index = sat_labels['Sat_indicator'] + " - " + sat_labels['Sat_unit']

    YQ.columns = labels['fd_lab'].dropna()
    YQ.index = sat_labels['Sat_indicator'] + " - " + sat_labels['Sat_unit']

    YQ.columns = pd.MultiIndex.from_tuples([split_string_new(c) for c in YQ.columns], names=['Country', 'Detail'])
    cols_to_drop = YQ.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = YQ.columns[cols_to_drop]
    YQ = YQ.drop(columns=cols_to_drop)
    YQ.columns = YQ.columns.remove_unused_levels()
    YQ.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country), detail) for country, detail in YQ.columns])
    YQ = YQ.groupby(level=[0, 1], axis=1).sum()

    ZQ = TQ[industry_labs]  # satellite accounts
    ZQ.columns = pd.MultiIndex.from_tuples([split_string_new(c) for c in ZQ.columns], names=['Country', 'Industry'])
    cols_to_drop = ZQ.columns.get_level_values('Country').isin(list_countries_drop)
    cols_to_drop = ZQ.columns[cols_to_drop]
    ZQ = ZQ.drop(columns=cols_to_drop)
    ZQ.columns = ZQ.columns.remove_unused_levels()
    ZQ.columns = pd.MultiIndex.from_tuples([(map_country_to_region(country), industry) for country, industry in ZQ.columns])
    ZQ = ZQ.groupby(level=[0, 1], axis=1).sum()
    ZQ.columns = [' - '.join(col) for col in ZQ.columns]
    ZQ.columns = [c.replace(' industry', '') for c in ZQ.columns]

    # We  extract CO2 emissions from intermediate consumptions
    system = 'OECD'
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

    emissions_detail = ZQ.loc[ZQ.index.str.contains(system),:]
    emissions_detail = emissions_detail.T
    emissions_detail = emissions_detail.loc[:, ~emissions_detail.columns.duplicated()]
    new_order = [f"'GHG_total_{system}_consistent' - kilotonnes CO2-equivalent", f"'co2_excl_short_cycle_org_c_total_{system}_consistent' - kilotonnes", f"'ch4_total_{system}_consistent' - kilotonnes", f"'n2o_total_{system}_consistent' - kilotonnes"]
    rest = [c for c in emissions_detail.columns if c not in new_order]
    new_order = new_order + rest
    emissions_detail = emissions_detail.reindex(columns = new_order)

    list_emissions_energy = ['1A1a', '1A1bc', '1A1b', '1A1ci', '1A1cii', '1A2', '1A2a', '1A2b', '1A2c', '1A2d', '1A3e', '1A2f',
                             '1A2g', '1A2h', '1A2i', '1A2j', '1A2k', '1A2l', '1A2m', '1A3b', '1A3b_noRES', '1A3b_RES', '1A3a', '1A3c', '1A3d', '1A3e', '1A4', '1A4a', '1A4b', '1A4ci',
                             '1A4cii', '1A4ciii', '1A5']
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
    # We calculate approximately the share of emissions associated to CO2 or CH4
    emissions_total = emissions_detail[emissions_detail.columns[emissions_detail.columns.str.contains('total')]]
    emissions_total = emissions_total[
        emissions_total.columns[emissions_total.columns.str.contains('|'.join(['GHG', 'co2', 'ch4']))]]
    emissions_total = emissions_total.div(emissions_total["'GHG_total_OECD_consistent' - kilotonnes CO2-equivalent"], axis=0)
    emissions_total["co2_excl_short_cycle_org_c"] = emissions_total["'co2_excl_short_cycle_org_c_total_OECD_consistent' - kilotonnes"]
    emissions_total["ch4"] = 1 - emissions_total["'co2_excl_short_cycle_org_c_total_OECD_consistent' - kilotonnes"]
    emissions_total = emissions_total[["co2_excl_short_cycle_org_c", "ch4"]]

    # emissions_energy = emissions_detail[emissions_detail.columns[emissions_detail.columns.str.contains('|'.join(list_emissions_energy))]]
    emissions_energy.columns = ['_'.join(''.join(idx.split("'")[1:]).split(' - ')) for idx in emissions_energy.columns]

    # new_index = pd.MultiIndex.from_tuples([('_'.join(idx.split('_')[:-3][:-1]), idx.split('_')[:-3][-1]) for idx in emissions_energy.columns])
    new_index = create_new_index(emissions_energy.columns)
    emissions_energy.columns = new_index

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
    energy_related_emissions = energy_related_emissions.fillna(0)
    # We calculate approximately overall the share of GHG emissions which can be associated to energy-related emissions
    energy_related_emissions = energy_related_emissions[['co2_excl_short_cycle_org_c', 'ch4']]
    energy_related_emissions = (energy_related_emissions * emissions_total).sum(axis=1)

    emissions_total_Z = pd.concat([emissions_total_Z, energy_related_emissions.rename("share_energy_related")], axis=1)

    # Approximation: final demand emissions is just CO2 emissions
    emissions_total_y = YQ.loc[YQ.index.str.contains('co2_excl_short_cycle_org'), :]
    emissions_total_y = emissions_total_y.loc[emissions_total_y.index.str.contains(system), :]
    # we allocate emissions to consumption of a given sector. This is a bit arbitrary probably, notably for the distribution of gaseous fuels through mains
    emissions_total_y = emissions_total_y.rename(index={
        f"'co2_excl_short_cycle_org_c_1A3b_noRES_{system}_consistent' - kilotonnes": 'Refined petroleum products',
        f"'co2_excl_short_cycle_org_c_1A4_{system}_consistent' - kilotonnes": 'Distribution of gaseous fuels through mains'  # this is a strong hypothesis, as many emissions from heating may come from other energy sources
    })
    emissions_total_y = emissions_total_y.loc[['Refined petroleum products', 'Distribution of gaseous fuels through mains'],:]
    emissions_total_y = emissions_total_y.groupby(level=0, axis=1).sum()
    sectors_to_add = pd.Index([e.split(' - ')[1] for e in emissions_total_Z.index])
    sectors_to_add = sectors_to_add.difference(emissions_total_y.index)
    additional_rows = pd.DataFrame(0, index=sectors_to_add, columns=emissions_total_y.columns)
    emissions_total_y = pd.concat([emissions_total_y, additional_rows], axis=0)  # we add other rows to match the shape of emissions_total_Z
    emissions_total_y = emissions_total_y.stack().reset_index()
    emissions_total_y.columns = ['sector', 'country', f"GHG_total_{system}_consistent_ktco2eq"]
    emissions_total_y['index'] = emissions_total_y['country'] + ' - ' + emissions_total_y['sector']
    emissions_total_y = emissions_total_y.set_index('index')[f"GHG_total_{system}_consistent_ktco2eq"].to_frame()  # be careful, here the country corresponds to the consumption of the final sector

    emissions_total = pd.concat([emissions_total_Z, emissions_total_y], axis=0)

    data_path = "GLORIA_MRIOs_57_2014/"

    country = country.lower()
    if len(country.split(' ')) > 1:
        country = '_'.join(country.split(' '))  # we process the name of the country for saving it
    #
    Z.to_pickle(Path(data_path) / Path(f"Z_{country}_RoW_2014.pkl"))
    Y.to_pickle(Path(data_path) / Path(f"Y_{country}_RoW_2014.pkl"))
    V.to_pickle(Path(data_path) / Path(f"V_{country}_RoW_2014.pkl"))
    emissions_total_Z.to_pickle(Path(data_path) / Path(f"emissions_Z_{country}_RoW_2014.pkl"))
    emissions_total_y.to_pickle(Path(data_path) / Path(f"emissions_Y_{country}_RoW_2014.pkl"))
    #
    # Save other objects
    with open(Path(data_path) / Path("sectors.pkl"), 'wb') as f:
        pickle.dump(sectors, f)
