## Copyright: Hubert Massoni

import pandas as pd
import country_converter as coco
import pycountry
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

def replace_last_occurrence(s, target, replacement):
    """Replaces the last occurence of target in string s. We uses complex reversing techniques, because
    there is easy solution to do so in python"""
    return s[::-1].replace(target[::-1], replacement[::-1], 1)[::-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GLORIA database.')
    parser.add_argument("--country", type=str, default='France', help="Country to do the processing")
    args = parser.parse_args()
    country = args.country  # we select the config we are interested in

    # Read CSV files
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

    # Assuming 'region' is a DataFrame with a column 'Region_acronyms'

    # Convert ISO3 to UN region names and handle exceptions
    region['un_cat'] = region['Region_acronyms'].apply(lambda x: coco.convert(names=x, to='UNregion') if x not in ['XAF', 'XAM', 'XAS', 'XEU', 'SDS', 'DYE'] else None)

    # # Assuming region is a DataFrame with 'Region_acronyms' column
    # region['un_cat'] = region['Region_acronyms'].apply(lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None)
    #
    region.loc[region['Region_acronyms'] == 'XAF', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'XAM', 'un_cat'] = 'Americas'
    region.loc[region['Region_acronyms'] == 'XAS', 'un_cat'] = 'Asia'
    region.loc[region['Region_acronyms'] == 'XEU', 'un_cat'] = 'Europe'
    region.loc[region['Region_acronyms'] == 'SDS', 'un_cat'] = 'Africa'
    region.loc[region['Region_acronyms'] == 'DYE', 'un_cat'] = 'Asia'
    #
    # # Convert ISO3 to EU membership
    # region['eu'] = region['Region_acronyms'].apply(lambda x: 'EU' if coco.convert(names=x, to='EU') else 'non-EU')
    #
    #
    # # Adding original labels to the data
    T.columns = labels['io_lab']
    T.index = labels['io_lab']
    Y.columns = labels['fd_lab'].dropna()
    Y.index = labels['io_lab']
    V.columns = labels['io_lab']
    V.index = labels['va_lab'].dropna()
    #
    # Create vectors for industry and product labels
    industry_labs = labels[labels['io_lab'].str.contains("industry")]['io_lab']
    product_labs = labels[~labels['io_lab'].isin(industry_labs)]['io_lab']
    #
    # # Extracting the use matrix, final demand matrix, value added matrix, and satellite accounts
    Z = T.loc[product_labs, industry_labs]  # use matrix
    Y = Y.loc[product_labs]  # final demand matrix
    V = V[industry_labs]  # value added matrix

    # Extracting France information
    V.index = pd.MultiIndex.from_tuples([split_string(i) for i in V.index], names=['Country', 'Detail'])
    V.columns = pd.MultiIndex.from_tuples([split_string(c) for c in V.columns], names=['Country', 'Industry'])

    V_country = V.loc[V.index.get_level_values('Country') == country, V.columns.get_level_values('Country') == country]
    V_country.index = V_country.index.droplevel(0)
    V_country.columns = V_country.columns.droplevel(0)
    # Attention, on n'a pas le secteur Growing beverages (coffee) pour la France
    V_country.columns = [c.replace(' industry', '') for c in V_country.columns]


    Y.index = pd.MultiIndex.from_tuples([split_string(i) for i in Y.index], names=['Country', 'Detail'])
    Y.columns = pd.MultiIndex.from_tuples([split_string(c) for c in Y.columns], names=['Country', 'Industry'])

    Y_country = Y.loc[Y.index.get_level_values('Country') == country, Y.columns.get_level_values('Country') == country]
    Y_country.index = Y_country.index.droplevel(0)
    Y_country.columns = Y_country.columns.droplevel(0)
    # Y_country.index = [i.replace(' product', '') for i in Y_country.index]
    Y_country.index = [replace_last_occurrence(i, ' product', '') for i in Y_country.index]

    Z.index = pd.MultiIndex.from_tuples([split_string(i) for i in Z.index], names=['Country', 'Detail'])
    Z.columns = pd.MultiIndex.from_tuples([split_string(c) for c in Z.columns], names=['Country', 'Industry'])

    Z_country = Z.loc[Z.index.get_level_values('Country') == country, Z.columns.get_level_values('Country') == country]
    Z_country.index = Z_country.index.droplevel(0)
    Z_country.columns = Z_country.columns.droplevel(0)
    Z_country.index = [replace_last_occurrence(i, ' product', '') for i in Z_country.index]
    Z_country.columns = [c.replace(' industry', '') for c in Z_country.columns]

    # Sum of columns for 'Exports' - sum all non-France columns
    exports_sum = Z.loc[:, Z.columns.get_level_values('Country') != country].sum(axis=1)
    exports_sum = exports_sum.loc[exports_sum.index.get_level_values('Country') == country]
    exports_sum.index = exports_sum.index.droplevel(0)
    exports_sum.index = [replace_last_occurrence(i, ' product', '') for i in exports_sum.index]
    exports_sum.name = 'Exports'

    Y_country = pd.concat([Y_country, exports_sum], axis=1)  # we add exports to the final demand for France

    # Sum of rows for 'Imports' - sum all non-France rows
    imports_sum = Z.loc[Z.index.get_level_values('Country') != country].sum(axis=0)
    imports_sum = imports_sum.loc[imports_sum.index.get_level_values('Country') == country]
    imports_sum.index = imports_sum.index.droplevel(0)
    imports_sum.index = [i.replace(' industry', '') for i in imports_sum.index]
    imports_sum.name = 'Imports'

    Y_country = pd.concat([Y_country, imports_sum], axis=1)

    del Y, V, Z
    gc.collect()

    #



    # # Read Excel sheets
    # region = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Regions")
    # sectors = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sectors")
    # labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Sequential region-sector labels")
    # sat_labels = pd.read_excel("GLORIA_MRIOs_57_2014/GLORIA_ReadMe_057.xlsx", sheet_name="Satellites")
    # #
    # # Create abbreviation of sector names (example for a few sectors)
    # sectors['sec_short'] = ["Wheat","Maize","CerOth","Legume","Rice","Veget","Sugar","Tobac","Fibre","CropOth","Grape","FruitNut","CropBev", "SpicePhar","Seed",
    #                         "Cattle","Sheep","Pig","Poultry","AnimOth","Forest","Fish","Crust","Coal","Lignite","Petrol", "Gas","IronOres","UranOres","AluOres",
    #                         "CopperOres","GoldOres","LedZinSilOres","NickelOres","TinOres","NonferOthOres", "StoneSand","ChemFert","Salt","OthServ","BeefMeat",
    #                         "SheepMeat","PorkMeat","PoultryMeat","MeatOth","FishProd","Cereal", "VegetProd","Fruit","FoodOth","Sweets","FatAnim","FatVeget","Dairy",
    #                         "Beverage","TobacProd","Textile","Leather","Sawmill", "Paper","Print","Coke","Petro","NFert","OthFert","ChemPetro","ChemInorg","ChemOrg",
    #                         "Pharma","ChemOth","Rubber","Plastic", "Clay","Ceramics","Cement","MinOth","IronSteel","Alu","Copper","Gold","LedZinSil","Nickel","Tin",
    #                         "NonferOth","FabMetal", "Machine","Vehicle","TransOth","Repair","Electronic","Electrical","FurnOth","Power","FuelDist","Water","Waste",
    #                         "Recovery", "ConstBuild","ConstCivil","TradeRep","Road","Rail","Pipe","WaterTrans","Air","Services","Post","Hospitality","Publishing",
    #                         "Telecom","Info","Finance","Real","ProfSci","Admin","Public","Edu","Health","Arts","Oth"]
    #
    # # Rename column names in labels DataFrame
    # labels.columns = ["Lfd_Nr", "io_lab", "fd_lab", "va_lab"]
    #
    # # Create vectors for industry and product labels
    # industry_labs = labels[labels['io_lab'].str.contains("industry")]['io_lab']
    # product_labs = labels[~labels['io_lab'].isin(industry_labs)]['io_lab']


    TQ = pd.read_csv("GLORIA_MRIOs_57_2014/20230727_120secMother_AllCountries_002_TQ-Results_2014_057_Markup001(full).csv", header=None)  # satellite accounts, disaggregated
    # YQ = pd.read_csv("GLORIA_MRIOs_57_2014/20230727_120secMother_AllCountries_002_YQ-Results_2014_057_Markup001(full).csv", header=None)  # satellite accounts, country level

    TQ.columns = labels['io_lab']
    TQ.index = sat_labels['Sat_indicator'] + " - " + sat_labels['Sat_unit']

    ZQ = TQ[industry_labs]  # satellite accounts

    # ZQ.index = pd.MultiIndex.from_tuples([split_string(i) for i in ZQ.index], names=['Country', 'Detail'])
    ZQ.columns = pd.MultiIndex.from_tuples([split_string(c) for c in ZQ.columns], names=['Country', 'Industry'])
    ZQ_country = ZQ.loc[:, ZQ.columns.get_level_values('Country') == country]

    # ZQ_country.index = ZQ_country.index.droplevel(0)
    ZQ_country.columns = ZQ_country.columns.droplevel(0)

    ZQ_country.columns = [c.replace(' industry', '') for c in ZQ_country.columns]

    emissions_country = ZQ_country.iloc[1841:1914, :]
    emissions_country = emissions_country.T

    emissions_total = ZQ_country.loc[ZQ_country.index.str.contains('total'),:]
    emissions_total = emissions_total.loc[emissions_total.index.str.contains('EDGAR'), :]
    emissions_total = emissions_total.T
    emissions_total = emissions_total.loc[:, ~emissions_total.columns.duplicated()]
    new_order = ["'GHG_total_EDGAR_consistent' - kilotonnes CO2-equivalent", "'co2_excl_short_cycle_org_c_total_EDGAR_consistent' - kilotonnes", "'ch4_total_EDGAR_consistent' - kilotonnes", "'n2o_total_EDGAR_consistent' - kilotonnes"]
    rest = [c for c in emissions_total.columns if c not in new_order]
    new_order = new_order + rest
    emissions_total = emissions_total.reindex(columns = new_order)
    emissions_total = emissions_total.rename(columns={
        "'GHG_total_EDGAR_consistent' - kilotonnes CO2-equivalent": "GHG_total_EDGAR_consistent_ktco2eq"
    })
    emissions_total.columns = [emissions_total.columns[0]] + ['_'.join(''.join(idx.split("'")[1:]).split(' - ')) for idx in emissions_total.columns[1:]]

    data_path = "GLORIA_MRIOs_57_2014/"

    country = country.lower()
    if len(country.split(' ')) > 1:
        country = '_'.join(country.split(' '))  # we process the name of the country for saving it

    Z_country.to_pickle(Path(data_path) / Path(f"Z_{country}.pkl"))
    Y_country.to_pickle(Path(data_path) / Path(f"Y_{country}.pkl"))
    V_country.to_pickle(Path(data_path) / Path(f"V_{country}.pkl"))
    emissions_total.to_pickle(Path(data_path) / Path(f"emissions_{country}.pkl"))
    #
    # Save other objects
    with open(Path(data_path) / Path("sectors.pkl"), 'wb') as f:
        pickle.dump(sectors, f)


    #
    # # Load concordance matrix with WIOD sectors = not necessary in my case
    # conv_df = pd.read_excel("Data/GLORIA_data/Gloria_wiod_concordance.xlsx", sheet_name="Concordance vector - energy")
    # conv_vec = conv_df['new_conv_sec'].tolist()
    #
    # # Generate new labels combining countries with converted sectors
    # countries = region['Region_acronyms'].tolist()
    # gloria_new_labs = [f"{country}_{sector}" for country in countries for sector in conv_vec]
    #
    # # Adjust dimensions and labels of matrices
    # Z.columns = gloria_new_labs
    # Z.index = gloria_new_labs
    # Y.columns = gloria_new_labs  # Adjust if Y has different column structure
    # Y.index = gloria_new_labs
    # V.columns = gloria_new_labs
    # V.index = gloria_new_labs
    # TQ.columns = gloria_new_labs
    # TQ.index = gloria_new_labs
    #
    # # Aggregate the matrices
    # Z_agg = Z.groupby(Z.index).sum().transpose().groupby(Z.columns).sum().transpose()
    # Y_agg = Y.groupby(Y.index).sum()
    # V_agg = V.T.groupby(V.columns).sum().transpose()
    # ZQ_agg = ZQ.T.groupby(ZQ.columns).sum().transpose()
    #
    # # Create new labels
    # countries = region['Region_acronyms'].tolist()
    # conv_vec_unique = list(set(conv_vec))
    # new_labs = [f"{country}_{sector}" for country in countries for sector in conv_vec_unique]
    #
    # # Reorder the matrices
    # Z_agg = Z_agg.loc[new_labs, new_labs]
    # Y_agg = Y_agg.loc[new_labs]
    # V_agg = V_agg[new_labs]
    # ZQ_agg = ZQ_agg[new_labs]