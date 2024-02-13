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

# ############### POST PROCESSING ######################
#
# Define data path
data_path = "GLORIA_MRIOs_57_2014/"

country = 'france'

# Read the data
Z = pd.read_pickle(f"{data_path}/Z_{country}.pkl")
Y = pd.read_pickle(f"{data_path}/Y_{country}.pkl")
V = pd.read_pickle(f"{data_path}/V_{country}.pkl")
emissions = pd.read_pickle(f"{data_path}/emissions_{country}.pkl")
# ZQ = pd.read_pickle(f"{data_path}/ZQ.pkl")
# region = pd.read_pickle(f"{data_path}/region.pkl")
sectors = pd.read_pickle(f"{data_path}/sectors.pkl")
#
# countries = region['Region_acronyms'].tolist()
# # Adjust the following line as per the scale of the data
# # countries = region['un_cat'].unique()  # For mid-scale data
# # countries = region['eu'].unique()      # For small data
#
# country_sec = [f"{country}_{sector}" for country in countries for sector in sectors]
#
# # Compute baseline CO2
# co2_unadj = ZQ.iloc[1843:1877, :].sum()
# x_unadj = Z.sum() + V.sum()
#
# # Baseline emission intensity
# d_unadj = co2_unadj / x_unadj
# d_unadj.fillna(0, inplace=True)
#
#
# # Replace zeros with a minimum value
# Z[Z == 0] = 0.0001
# Y[Y == 0] = 0.0001
# V[V == 0] = 0.0001
#
# # # Mirroring procedure for handling negative inventories
Y_inv = Y[Y.columns[Y.columns.str.contains('inv')]]
Y_inv[Y_inv > 0] = 0
Y_inv_total = Y_inv.sum(axis=1)  # sum over columns

Y[Y < 0] = 0  # get rid of negative inventories
# add a row to V which is the sum of all rows in V
V.loc['va'] = V.sum(axis=0)
if not V.columns.equals(Y_inv_total.index):
    # Aligning indices if they are not the same
    V, Y_inv_total = V.align(Y_inv_total, axis=1, copy=False)

V.loc['va'] = V.loc['va'] + abs(Y_inv_total)

# Handle some problems in the added value, specific to France case I think
# replace negative values in V['va'] by the value V['Compensation of employees D.1']
V.loc['va'][V.loc['va'] < 0] = V.loc['Compensation of employees D.1'][V.loc['va'] < 0]
# when V.loc['va'] is smaller than V.loc['Compensation of employees D.1'], replace by the latter, so that the share of labor is never larger than 1
V.loc['va'][V.loc['va'] < V.loc['Compensation of employees D.1']] = V.loc['Compensation of employees D.1'][V.loc['va'] < V.loc['Compensation of employees D.1']]

#
#
# #### Create complete IO matrix to match notations from Grassi
# # Step 1: Rename Indices and Columns
sector_tuples = list(zip(sectors['sec_short'], sectors['Sector_names']))
sector_dict = dict(zip(sectors['Sector_names'], sector_tuples))
country_dict = {
    'France': 'FRA',
    'RoW': 'RoW',
    'United States of America': 'USA',
    'Denmark': 'DNK',
    'Germany': 'DEU'
}
#
finaldemand_dict = {
    'Household final consumption P.3h': ('HHConsumption', 'Household final consumption P.3h'),
    'Non-profit institutions serving households P.3n': ('NPconsumption', 'Non-profit institutions serving households P.3n'),
    'Government final consumption P.3g': ('GovConsumption', 'Government final consumption P.3g'),
    'Gross fixed capital formation P.51': ('GFCP', 'Gross fixed capital formation P.51'),
    'Changes in inventories P.52': ('Inventory', 'Changes in inventories P.52'),
    'Acquisitions less disposals of valuables P.53' : ('Acquisitions', 'Acquisitions less disposals of valuables P.53'),
    'Exports': ('Exports', 'Exports'),
    'Imports': ('Imports', 'Imports'),
    'Total final use': ('Total', 'Total final use')
}
#
valueadded_dict = {
    'Compensation of employees D.1': ('EmpComp', 'Compensation of employees D.1'),
    'Taxes on production D.29': ('Taxes', 'Taxes on production D.29'),
    'Subsidies on production D.39': ('Subsidies', 'Subsidies on production D.39'),
    'Net operating surplus B.2n': ('NetOP', 'Net operating surplus B.2n'),
    'Net mixed income B.3n': ('NetMI', 'Net mixed income B.3n'),
    'Consumption of fixed capital K.1': ('CFC', 'Consumption of fixed capital K.1'),
    'va': ('ValueAdded', 'ValueAdded'),
}
#
Z_multiindex = pd.MultiIndex.from_tuples([sector_dict[i] for i in Z.index], names=['Code', 'Description'])
Z.columns = pd.MultiIndex.from_tuples([sector_dict[c] for c in Z.columns], names=['Code', 'Description'])
Z.index = Z_multiindex

V.index = pd.MultiIndex.from_tuples([valueadded_dict[i] for i in V.index], names=['Code', 'Description'])
V.columns = pd.MultiIndex.from_tuples([sector_dict[c] for c in V.columns], names=['Code', 'Description'])

Y['Imports'] = - Y['Imports']  # convention to have imports displayed as negative values
Y['Total final use'] = Y.sum(axis=1)
Y.index = pd.MultiIndex.from_tuples([sector_dict[i] for i in Y.index], names=['Code', 'Description'])
Y.columns = pd.MultiIndex.from_tuples([finaldemand_dict[c] for c in Y.columns], names=['Code', 'Description'])

emissions.index = pd.MultiIndex.from_tuples([sector_dict[i] for i in emissions.index], names=['Code', 'Description'])

#
# Step 2: Aggregate DataFrames
# 2.1 Add va (from V) as additional rows to Z to create tmp
tmp = pd.concat([Z, V], axis=0)

#
#
additional_rows = pd.DataFrame(0, index=tmp.index.difference(Y.index), columns=Y.columns)
Y_aligned = pd.concat([Y, additional_rows], axis=0)
#
final_io_table = pd.concat([tmp, Y_aligned], axis=1)

# final_io_table.index = final_io_table.index.droplevel(1)
# final_io_table.columns = final_io_table.columns.droplevel(1)
# final_io_table.loc['labor_share'] = final_io_table.loc['EmpComp'] / final_io_table.loc['ValueAdded']

# Y.xs('HHConsumption', level=0, axis=1).sum()
# Y['share'] = Y.xs('HHConsumption', level=0, axis=1) / Y.xs('HHConsumption', level=0, axis=1).sum()
# Y = Y.sort_values(by="share", ascending=False)


final_io_table.to_excel(f'GLORIA_MRIOs_57_2014/{country}_IO_table_2014.xlsx', sheet_name='IO_table')
emissions.to_excel(f'GLORIA_MRIOs_57_2014/{country}_emissions_2014.xlsx', sheet_name='emissions')

#
# # Check for negative value added
# has_negative_va = (va < 0).any()
# # Comparison with original V sum, if needed
# is_equal_to_original = va.equals(V.sum(axis=1))
#
# # Total output
# x = Z.sum(axis=1) + va
#
# # Value added share
# v_share = va / x
# v_share.index = country_sec
#
# # Technical coefficient matrix
# A = Z.div(x, axis=0)
#
# # Leontief Inverse
# I = np.identity(len(A))
# L = inv(I - A)
#
# # Output allocation matrix and Ghosh Inverse
# B = Z.div(x, axis=1)
# G = inv(I - B)
#
# # Aggregate final demand categories
# sum_c_categories = pd.DataFrame(np.tile(np.identity(len(countries)), (6, 1)))
# c_si = Y.dot(sum_c_categories.T)
# c_si.index = country_sec
# c_si.columns = countries
#
# # Total expenditures of consumers in each country
# C = c_si.sum(axis=0)
#
# # Aggregate final demand vector across destinations
# c = c_si.sum(axis=1)
#
# # Expenditure shares
# G_si = c_si.div(C, axis=1)
#
# # Gloria CO2
# co2 = d_unadj * x
#
# # Direct emission intensities
# d = co2 / x
#
# # Total emission intensities of each sector
# e = np.dot(L, d)
# e.index = country_sec