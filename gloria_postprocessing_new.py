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
import os
import pickle
from numpy.linalg import inv
import numpy as np
import re
import gc
from pathlib import Path

"""Postprocessing module
This module performs some postprocessing on the input-output table, the final demand table, the value-added table, and the emissions tables.
"""



country_dict = {
    'France': 'FRA',
    'RoW': 'ROW',
    'United States of America': 'USA',
    'Denmark': 'DNK',
    'Germany': 'DEU',
    'Spain': 'SPA',
    'Europe': 'EUR'
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
    'Total final use': ('TotalUse', 'Total final use')
}
#
v_dict = {
    'Compensation of employees D.1': ('EmpComp', 'Compensation of employees D.1'),
    'Taxes on production D.29': ('Taxes', 'Taxes on production D.29'),
    'Subsidies on production D.39': ('Subsidies', 'Subsidies on production D.39'),
    'Net operating surplus B.2n': ('NetOP', 'Net operating surplus B.2n'),
    'Net mixed income B.3n': ('NetMI', 'Net mixed income B.3n'),
    'Consumption of fixed capital K.1': ('CFC', 'Consumption of fixed capital K.1'),
    'va': ('ValueAdded', 'ValueAdded'),
}


def sort_multiindex(df, sector_order, axis=0):
    """Function to sort the dataframe based on a given order of sectors."""
    if axis==0:  # we sort according to index
        # Extract the sector code from the MultiIndex
        sector_codes = [idx.split(' - ')[1].strip() for idx in df.index.get_level_values('Description')]

        # Get the order for each sector in the MultiIndex
        sector_orders = [sector_order.get(code, len(sector_order)) for code in sector_codes]

        # Create a new MultiIndex with the sector order
        new_index = pd.MultiIndex.from_arrays([df.index.get_level_values('Code'), df.index.get_level_values('Description'), sector_orders],
                                              names=['Code', 'Description', 'Order'])

        # Set the new MultiIndex
        df.index = new_index

        # Sort by the sector order and drop the order level
        df = df.sort_values(by='Order', axis=0).droplevel('Order')

    else:  # we sort according to column
        sector_codes = [idx.split(' - ')[1].strip() for idx in df.columns.get_level_values('Description')]

        # Get the order for each sector in the MultiIndex
        sector_orders = [sector_order.get(code, len(sector_order)) for code in sector_codes]

        # Create a new MultiIndex with the sector order
        new_index = pd.MultiIndex.from_arrays([df.columns.get_level_values('Code'), df.columns.get_level_values('Description'), sector_orders],
            names=['Code', 'Description', 'Order'])

        # Set the new MultiIndex
        df.columns = new_index

        # Sort by the sector order and drop the order level
        df = df.sort_values(by='Order', axis=1).droplevel('Order', axis=1)
    return df

def sort_columns(columns, country, nb=2):
    """Function sorting sectors so that the domestic country appears first."""
    # TODO: function to be modified if I have more than 2 countries
    sorted_columns = []
    rev = False
    if country[0] >= 'r':  # check the first letter of the country, to sort accordingly
        rev = True
    for i in range(0, len(columns), nb):
        pair = sorted(columns[i:i + nb], reverse=rev)
        sorted_columns.extend(pair)
    new_index = pd.MultiIndex.from_tuples(sorted_columns)
    return new_index


if __name__ == '__main__':
    # ############### POST PROCESSING ######################

    # Define inputs path
    data_path = "GLORIA_MRIOs_57_2014"

    country = 'europe'

    # Read the inputs
    Z = pd.read_pickle(f"{data_path}/Z_{country}_RoW_2014.pkl")
    Y = pd.read_pickle(f"{data_path}/Y_{country}_RoW_2014.pkl")
    V = pd.read_pickle(f"{data_path}/V_{country}_RoW_2014.pkl")
    emissions_Z = pd.read_pickle(f"{data_path}/emissions_Z_{country}_RoW_2014.pkl")
    emissions_Y = pd.read_pickle(f"{data_path}/emissions_Y_{country}_RoW_2014.pkl")

    # ZQ = pd.read_pickle(f"{data_path}/ZQ.pkl")
    # region = pd.read_pickle(f"{data_path}/region.pkl")
    sectors = pd.read_pickle(f"{data_path}/sectors.pkl")

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

    sector_order = {name: i for i, name in enumerate(sectors['Sector_names'])}  # get prespecified order of sectors

    Z.index = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in Z.index], names=['Code', 'Description'])
    Z.columns = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in Z.columns], names=['Code', 'Description'])

    V.index = pd.MultiIndex.from_tuples([v_dict[i] for i in V.index], names=['Code', 'Description'])
    V.columns = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in V.columns], names=['Code', 'Description'])

    countries = [entry.split(' - ')[0] for entry in Y.columns]
    grouped = Y.groupby(countries, axis=1).sum()
    grouped.columns = [f'{col} - Total final use' for col in grouped.columns]
    Y = pd.concat([Y, grouped], axis=1)

    Y.index = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in Y.index], names=['Code', 'Description'])
    Y.columns =  pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + finaldemand_dict[entry.split(' - ')[1]][0], entry) for entry in Y.columns], names=['Code', 'Description'])

    emissions_Z.index = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in emissions_Z.index], names=['Code', 'Description'])
    emissions_Y.index = pd.MultiIndex.from_tuples([(country_dict[entry.split(' - ')[0]] + sector_dict[entry.split(' - ')[1]][0], entry) for entry in emissions_Y.index], names=['Code', 'Description'])

    # sorting dataframe to ensure they are in the right order
    Y = sort_multiindex(Y, sector_order, axis=0)
    V = sort_multiindex(V, sector_order, axis=1)
    Z = sort_multiindex(Z, sector_order, axis=0)
    Z = sort_multiindex(Z, sector_order, axis=1)
    emissions_Z = sort_multiindex(emissions_Z, sector_order, axis=0)
    emissions_Y = sort_multiindex(emissions_Y, sector_order, axis=0)

    sorted_columns = sort_columns(Z.columns, country=country, nb=2)
    sorted_index = sort_columns(Z.index, country=country, nb=2)
    Z = Z.reindex(index=sorted_index)
    Z = Z.reindex(columns=sorted_columns)

    sorted_columns = sort_columns(V.columns, country=country, nb=2)
    V = V.reindex(columns=sorted_columns)

    sorted_index = sort_columns(Y.index, country=country, nb=2)
    Y = Y.reindex(index=sorted_index)

    sorted_index = sort_columns(emissions_Z.index, country=country, nb=2)
    emissions_Z = emissions_Z.reindex(index=sorted_index)

    sorted_index = sort_columns(emissions_Y.index, country=country, nb=2)
    emissions_Y = emissions_Y.reindex(index=sorted_index)

    tmp = pd.concat([Z, V], axis=0)

    additional_rows = pd.DataFrame(0, index=tmp.index.difference(Y.index), columns=Y.columns)
    Y_aligned = pd.concat([Y, additional_rows], axis=0)

    final_io_table = pd.concat([tmp, Y_aligned], axis=1)

    # final_io_table.index = final_io_table.index.droplevel(1)
    # final_io_table.columns = final_io_table.columns.droplevel(1)
    # final_io_table.loc['labor_share'] = final_io_table.loc['EmpComp'] / final_io_table.loc['ValueAdded']

    # Y.xs('HHConsumption', level=0, axis=1).sum()
    # Y['share'] = Y.xs('HHConsumption', level=0, axis=1) / Y.xs('HHConsumption', level=0, axis=1).sum()
    # Y = Y.sort_values(by="share", ascending=False)

    final_io_table[final_io_table == 0] = 0.0001
    #
    final_io_table.to_excel(f'GLORIA_MRIOs_57_2014/{country}_RoW_IO_table_2014.xlsx', sheet_name='table')
    emissions_Z.to_excel(f'GLORIA_MRIOs_57_2014/{country}_RoW_emissions_Z_2014.xlsx', sheet_name='emissions')
    emissions_Y.to_excel(f'GLORIA_MRIOs_57_2014/{country}_RoW_emissions_Y_2014.xlsx', sheet_name='emissions')


