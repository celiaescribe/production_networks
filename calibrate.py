import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from utils import add_long_description, flatten_index, unflatten_index, unflatten_index_in_df, same_df


DIRTY_ENERGY_SECTORS = ['Refined Petroleum products', 'Distribution of gaseous fuels through mains', 'Hard coal', 'Lignite and peat',
                        'Petroleum extraction', 'Gas extraction', 'Coke oven products']  # those sectors are responsible for GHG emissions when they are burnt
DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']

DIRTY_ENERGY_USE = ['Petro', 'FuelDist']  # sectors related to consumption of dirty energy for households (different from the ones for production)

ENERGY_SECTORS = DIRTY_ENERGY_SECTORS +  ['Electric power generation, transmission and distribution']

@dataclass
class CalibOutput:
    sectors: pd.DataFrame
    emissions: pd.DataFrame
    xsi: pd.DataFrame
    psi: pd.DataFrame
    Omega: pd.DataFrame
    Gamma: pd.DataFrame
    Domestic: pd.DataFrame
    Delta: pd.DataFrame
    share_GNE: pd.Series
    sectors_dirty_energy: pd.DataFrame
    final_use_dirty_energy: pd.DataFrame
    descriptions: pd.Series

    def to_excel(self, path):
        with pd.ExcelWriter(path) as writer:
            for current_df, sheet_name in [
                (self.sectors, "sectors"),
                (self.emissions, "emissions"),
                (self.xsi, "xsi"),
                (self.psi, "psi"),
                (self.Omega, "Omega"),
                (self.Gamma, "Gamma"),
                (self.Domestic, "Domestic"),
                (self.Delta, "Delta"),
                (self.sectors_dirty_energy, "sectors_dirty_energy"),
                (self.final_use_dirty_energy, "final_use_dirty_energy"),
                (self.share_GNE, "share_GNE"),
                (self.descriptions, "descriptions"),
            ]:
                is_dataframe = type(current_df) is not pd.Series

                # Copy the dataframe to avoid modifying the original one
                df_to_write = current_df.copy()
                # Flatten the columns if dataframe
                if is_dataframe:
                    df_to_write.columns = flatten_index(df_to_write.columns)
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
            sectors = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="sectors", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            emissions = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="emissions", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            xsi = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="xsi", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            psi = pd.read_excel(xls, sheet_name="psi", index_col=0).drop(columns="long_description")
            psi.index.names = ["Sector"]
            Omega = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Omega", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            Gamma = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Gamma", index_col=0)
                .drop(columns="long_description"),
                axis=[0, 1], level_names=["Country", "Sector"]
            )
            Domestic = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Domestic", index_col=0)
                .drop(columns="long_description"),
                axis=[0, 1], level_names=["Country", "Sector"]
            )
            Delta = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Delta", index_col=0)
                .drop(columns="long_description"),
                axis=[0, 1], level_names=["Country", "Sector"]
            )
            share_GNE = pd.read_excel(xls, sheet_name="share_GNE", index_col=0, header=None).squeeze()
            share_GNE.index.names = ["Country"]
            sectors_dirty_energy = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="sectors_dirty_energy", index_col=0)
                .drop(columns="long_description"),
                axis=[0, 1], level_names=["Country", "Sector"])
            final_use_dirty_energy = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="final_use_dirty_energy", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            descriptions = pd.read_excel(xls, sheet_name="descriptions", index_col=0, header=None).squeeze()
            descriptions.index.name = "Sector"
            descriptions.name = 0
        return cls(sectors, emissions, xsi, psi, Omega, Gamma, Domestic, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy, descriptions)

    def equals(self, other):
        return (
                same_df(self.sectors, other.sectors)
                and same_df(self.emissions, other.emissions)
                and same_df(self.xsi, other.xsi)
                and same_df(self.psi, other.psi)
                and same_df(self.Omega, other.Omega)
                and same_df(self.Gamma, other.Gamma)
                and same_df(self.Domestic, other.Domestic)
                and same_df(self.Delta, other.Delta)
                and same_df(self.sectors_dirty_energy, other.sectors_dirty_energy)
                and same_df(self.final_use_dirty_energy, other.final_use_dirty_energy)
                and self.descriptions.equals(other.descriptions)
        )


def process_excel(file_path):
    logging.info("Reading data")
    df = pd.read_excel(file_path, index_col=0, header=0)
    # first row should be the second level of the multiindex for columns
    df.columns = pd.MultiIndex.from_arrays([df.columns, df.iloc[0]])
    df.index = pd.MultiIndex.from_arrays([df.index, df.iloc[:,0]])
    # remove first column
    df = df.iloc[2:, 1:]
    # Convert all DataFrame values to floats
    df = df.astype(float)

    value_added_names = ['EmpComp', 'CFC', 'NetMI', 'NetOP', 'Subsidies', 'Taxes', 'ValueAdded']
    final_use_names = ['Acquisitions', 'Inventory', 'GovConsumption', 'GFCP', 'HHConsumption', 'NPconsumption', 'TotalUse']

    long_description = df.columns.get_level_values(1)
    descriptions = {idx[0][3:]: idx[1].split(' - ')[1] for idx in df.columns}
    descriptions = {key: value for key, value in descriptions.items() if key not in final_use_names}
    descriptions = pd.Series(descriptions)
    descriptions.index.name = 'Sector'
    df.columns = df.columns.get_level_values(0)
    df.index = df.index.get_level_values(0)

    def split_codes_names(lst):
        country = [x[:3] for x in lst]
        sectors = [x[3:] for x in lst]
        return country, sectors

    df.columns = pd.MultiIndex.from_arrays(split_codes_names(df.columns))


    value_added = df.loc[df.index.get_level_values(0).isin(value_added_names)]
    value_added = value_added.loc[:, ~ value_added.columns.get_level_values(1).isin(final_use_names)]

    df = df.loc[~ df.index.get_level_values(0).isin(value_added_names)]
    df.index = pd.MultiIndex.from_arrays(split_codes_names(df.index))
    final_use = df.loc[:, df.columns.get_level_values(1).isin(final_use_names)]

    df = df.loc[:, ~ df.columns.get_level_values(1).isin(final_use_names)]

    df.index.names = ['Country', 'Sector']
    df.columns.names = ['Country', 'Sector']
    final_use.index.names = ['Country', 'Sector']
    final_use.columns.names = ['Country', 'Sector']
    value_added.columns.names = ['Country', 'Sector']

    final_use[final_use < 0] = 0
    df[df < 0] = 0
    df = df.T  # transposition: ligne i corresponds to what sector i uses.

    value_added.loc['wli_over_vhi'] = value_added.loc['EmpComp'] / value_added.loc['ValueAdded']

    final_use = final_use.xs('TotalUse', level=1, axis=1)  # get only final use

    return df, value_added, final_use, descriptions

def process_emissions(file_path_emissions_Z, file_path_emissions_Y):
    emissions_Z = pd.read_excel(file_path_emissions_Z, index_col=0, header=0)
    emissions_Z = emissions_Z.iloc[:,1:]
    emissions_Z = emissions_Z[['GHG_total_OECD_consistent_ktco2eq', 'share_energy_related']].rename(columns={'GHG_total_OECD_consistent_ktco2eq': 'total'})
    # emissions_Z = emissions_Z.iloc[:,0].to_frame(name='sectors')

    emissions_Y = pd.read_excel(file_path_emissions_Y, index_col=0, header=0)
    emissions_Y = emissions_Y.iloc[:,1:]
    emissions_Y = emissions_Y.iloc[:,0].to_frame(name='final_demand')
    # Convert all DataFrame values to floats
    emissions_Z = emissions_Z.astype(float)
    emissions_Y = emissions_Y.astype(float)

    def split_codes_names(lst):
        country = [x[:3] for x in lst]
        sectors = [x[3:] for x in lst]
        return country, sectors

    emissions_Z.index = pd.MultiIndex.from_arrays(split_codes_names(emissions_Z.index))
    emissions_Y.index = pd.MultiIndex.from_arrays(split_codes_names(emissions_Y.index))

    emissions_total = pd.concat([emissions_Z, emissions_Y], axis=1)
    emissions_total.index.names = ['Country', 'Sector']

    # get share of total emissions in the given country
    emissions_total = emissions_total.assign(share_emissions_total_sectors = emissions_total['total'].div(emissions_total[['total', 'final_demand']].sum(axis=1).groupby(level='Country').sum()))  # we estimate the share of domestic emissions coming from different sectors
    emissions_total = emissions_total.assign(share_emissions_total_finaldemand=emissions_total['final_demand'].div(emissions_total[['total', 'final_demand']].sum(axis=1).groupby(level='Country').sum()))

    # share_emissions = emissions_total.div(emissions_total.sum(axis=1).groupby(level='Country').sum(), axis=0)

    return emissions_total

def get_main_stats(df, final_use):
    logging.info("Main statistics")
    sectors = df.sum(axis=1).to_frame(name='pmXi')
    sectors['pyi'] = df.sum(axis=0) + final_use.sum(axis=1)

    Gamma = df.div(sectors['pyi'], axis=0)

    original_order = df.columns.get_level_values('Sector').unique()
    Omega = df.groupby(level='Sector', axis=1).sum()  # sum by sector
    Omega = Omega[original_order]
    Omega = Omega.div(sectors['pmXi'], axis=0)  # divide by total costs for each sector

    # share of intermediate consumption in dirty energy sectors
    sectors_dirty_energy = df.loc[:, df.columns.get_level_values('Sector').isin(DIRTY_ENERGY_SECTORS)]
    sectors_dirty_energy = sectors_dirty_energy.div(sectors_dirty_energy.sum(axis=1), axis=0)

    Domestic = df.div(df.groupby(level='Sector', axis=1).sum(), axis=0)
    # This could also be written as Domestic = df / df.groupby(level='Sector', axis=1).sum()

    # assert that the sum of Domestic, grouped by Sector  is equal to 1
    assert np.allclose(Domestic.groupby(level='Sector', axis=1).sum(), 1)

    psi = final_use.groupby('Sector').sum()  # sum of all final use
    psi = psi / psi.sum(axis=0)  # share of sector in final demand
    psi = psi.loc[original_order]
    xsi = final_use.div(final_use.groupby('Sector').sum())

    # share of final consumption in dirty energy sectors, for each of the countries
    final_use_dirty_energy = final_use.loc[final_use.index.get_level_values('Sector').isin(DIRTY_ENERGY_SECTORS), :]
    final_use_dirty_energy = final_use_dirty_energy / final_use_dirty_energy.sum(axis=0)

    sectors['eta'] = 1 - sectors['pmXi'] / sectors['pyi']  # share of added value in total output

    sectors['phi'] = final_use.sum(axis=1) / sectors['pyi']
    for country in final_use.columns:
        sectors[f'phi_{country}'] = final_use[country] / sectors['pyi']

    sectors['va'] = sectors['pyi'] - sectors['pmXi']  # we redefine va from other values, to ensure that data is correctly balanced

    sectors['gamma'] = value_added.loc['wli_over_vhi']

    Delta = df.div(sectors['pyi'], axis=1)  # Delta_ij =  expenditure on j by i as a share of total production of j

    empcomp = value_added.loc['EmpComp'].to_frame(name='share_labor')
    share_labor = empcomp.div(empcomp.groupby('Country').sum())
    # empcomp.squeeze() / empcomp.groupby('Country').sum().squeeze()   # other way to write that

    value_added.loc['Capital'] = value_added.loc['ValueAdded'] - value_added.loc['EmpComp']  # we define capital as the rest of the added value apart from labor
    assert (value_added.loc['Capital'] >= 0).all()
    capital = value_added.loc['Capital'].to_frame(name='share_capital')
    share_capital = capital.div(capital.groupby('Country').sum())

    sectors = pd.concat([sectors, share_labor, share_capital], axis=1)

    tmp = (sectors['gamma'] * sectors['va']).to_frame()
    sectors['rev_labor'] =  tmp.div(sectors['va'].groupby('Country').sum(), axis=0).rename(columns={0: 'rev_labor'})  # GNE estimated from final use

    tmp = ((1-sectors['gamma']) * sectors['va']).to_frame()
    sectors['rev_capital'] =  tmp.div(sectors['va'].groupby('Country').sum(), axis=0).rename(columns={0: 'rev_capital'})  # GNE estimated from final use

    total_GDP = sectors['va'].sum()
    assert total_GDP == sectors['va'].sum(), "Accounting equation is not verified for world GPD. It should be the same, whether calculated from value added or final use"
    share_GNE = sectors['va'].groupby('Country').sum() / total_GDP  # we rely on final_use to determine the share of each GNE in the total GDP (we could also have used value added)

    domestic_domar_weights = sectors['pyi'] / final_use.sum()
    sectors['domestic_domar_weights'] = domestic_domar_weights
    return sectors, Gamma, Omega, Domestic, psi, xsi, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy


def networks_stats(Gamma, col_final_use, total_output):
    index = Gamma.index
    out_degree = Gamma.sum(0)

    centrality, _, _, _ = np.linalg.lstsq(
        (np.eye(Gamma.shape[0]) - Gamma).T,
        col_final_use / col_final_use.sum(),
        rcond=None,
    )
    centrality = pd.Series(centrality, index=index)

    # Absurd notations. This einsum is equivalent to
    # (1 / (1 - 1/total_output)) * (G.T * total_output).sum(1)
    upstreamness = np.einsum(
        "i,ij,j -> i", 1 / (1 - 1 / total_output), Gamma.T, total_output
    )
    upstreamness = pd.Series(upstreamness, index=index)

    domar = total_output / col_final_use.sum()

    result = pd.DataFrame(
        {
            "out_degree": out_degree,
            "centrality": centrality,
            "upstreamness": upstreamness,
            "domar": domar,
        }
    )

    return result

if __name__ == '__main__':
    country = 'france'
    file_path = f'data_deep/{country}_RoW_IO_table_2014.xlsx'
    file_path_emissions_Z = f'data_deep/{country}_RoW_emissions_Z_2014.xlsx'
    file_path_emissions_Y = f'data_deep/{country}_RoW_emissions_Y_2014.xlsx'

    df, value_added, final_use, descriptions = process_excel(file_path)

    emissions_total = process_emissions(file_path_emissions_Z, file_path_emissions_Y)

    sectors, Gamma, Omega, Domestic, psi, xsi, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy = get_main_stats(df, final_use)
    calib = CalibOutput(sectors, emissions_total, xsi, psi, Omega, Gamma, Domestic, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy, descriptions)
    calib.to_excel(f"outputs/calib_{country}.xlsx")
    calib2 = CalibOutput.from_excel(f"outputs/calib_{country}.xlsx")
    assert calib.equals(calib2)

# networks_stats(Gamma, col_final_use, total_output)