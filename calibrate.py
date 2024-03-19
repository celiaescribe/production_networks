import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from utils import add_long_description, flatten_index, unflatten_index, unflatten_index_in_df, same_df


DIRTY_ENERGY_SECTORS = ['Coal', 'Lignite', 'Petrol', 'Gas', 'Coke', 'Petro', 'FuelDist']  # those sectors are responsible for GHG emissions when they are burnt

DIRTY_ENERGY_USE = ['Petro', 'FuelDist']  # sectors related to consumption of dirty energy for households (different from the ones for production)

ENERGY_SECTORS = DIRTY_ENERGY_SECTORS +  ['Power']

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
class CalibOutput:
    sectors: pd.DataFrame
    emissions: pd.DataFrame
    xsi: pd.DataFrame
    psi: pd.DataFrame
    costs_energy_final: pd.DataFrame
    psi_energy: pd.DataFrame
    psi_non_energy: pd.DataFrame
    costs_durable_final: pd.DataFrame
    psi_durable: pd.DataFrame
    psi_non_durable: pd.DataFrame
    Omega: pd.DataFrame
    costs_energy: pd.DataFrame
    Omega_energy: pd.DataFrame
    Omega_non_energy: pd.DataFrame
    Gamma: pd.DataFrame
    Leontieff: pd.DataFrame
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
                (self.costs_energy_final, "costs_energy_final"),
                (self.psi_energy, "psi_energy"),
                (self.psi_non_energy, "psi_non_energy"),
                (self.costs_durable_final, "costs_durable_final"),
                (self.psi_durable, "psi_durable"),
                (self.psi_non_durable, "psi_non_durable"),
                (self.Omega, "Omega"),
                (self.costs_energy, "costs_energy"),
                (self.Omega_energy, "Omega_energy"),
                (self.Omega_non_energy, "Omega_non_energy"),
                (self.Gamma, "Gamma"),
                (self.Leontieff, "Leontieff"),
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
            psi_energy = pd.read_excel(xls, sheet_name="psi_energy", index_col=0).drop(columns="long_description")
            psi_energy.index.names = ["Sector"]
            psi_non_energy = pd.read_excel(xls, sheet_name="psi_non_energy", index_col=0).drop(columns="long_description")
            psi_non_energy.index.names = ["Sector"]
            costs_energy_final = pd.read_excel(xls, sheet_name="costs_energy_final", index_col=0)
            # costs_energy_final = pd.read_excel(xls, sheet_name="costs_energy_final", index_col=0).drop(columns="long_description")
            # costs_energy_final.index.names = ["Sector"]
            psi_durable = pd.read_excel(xls, sheet_name="psi_durable", index_col=0).drop(columns="long_description")
            psi_durable.index.names = ["Sector"]
            psi_non_durable = pd.read_excel(xls, sheet_name="psi_non_durable", index_col=0).drop(columns="long_description")
            psi_non_durable.index.names = ["Sector"]
            costs_durable_final = pd.read_excel(xls, sheet_name="costs_durable_final", index_col=0)
            Omega = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Omega", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            costs_energy = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="costs_energy", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            Omega_energy = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Omega_energy", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            Omega_non_energy = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Omega_non_energy", index_col=0)
                .drop(columns="long_description"),
                axis=0, level_names=["Country", "Sector"])
            Gamma = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Gamma", index_col=0)
                .drop(columns="long_description"),
                axis=[0, 1], level_names=["Country", "Sector"]
            )
            Leontieff = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="Leontieff", index_col=0)
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
        return cls(sectors, emissions, xsi, psi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable,
                   psi_non_durable, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Domestic, Delta, share_GNE, sectors_dirty_energy,
                   final_use_dirty_energy, descriptions)

    def equals(self, other):
        return (
                same_df(self.sectors, other.sectors)
                and same_df(self.emissions, other.emissions)
                and same_df(self.xsi, other.xsi)
                and same_df(self.psi, other.psi)
                and same_df(self.costs_energy_final, other.costs_energy_final)
                and same_df(self.psi_energy, other.psi_energy)
                and same_df(self.psi_non_energy, other.psi_non_energy)
                and same_df(self.Omega, other.Omega)
                and same_df(self.costs_energy, other.costs_energy)
                and same_df(self.Omega_energy, other.Omega_energy)
                and same_df(self.Omega_non_energy, other.Omega_non_energy)
                and same_df(self.Gamma, other.Gamma)
                and same_df(self.Leontieff, other.Leontieff)
                and same_df(self.Domestic, other.Domestic)
                and same_df(self.Delta, other.Delta)
                and same_df(self.sectors_dirty_energy, other.sectors_dirty_energy)
                and same_df(self.final_use_dirty_energy, other.final_use_dirty_energy)
                and self.descriptions.equals(other.descriptions)
        )


def process_excel(file_path):
    """Read the excel file and process it to get the different dataframes, including IO table, value added and final use."""
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
    df = df.T  # transpose: lign i corresponds to what sector i uses.

    value_added.loc['wli_over_vhi'] = value_added.loc['EmpComp'] / value_added.loc['ValueAdded']

    final_use = final_use.xs('TotalUse', level=1, axis=1)  # get only final use

    return df, value_added, final_use, descriptions

def process_emissions(file_path_emissions_Z, file_path_emissions_Y):
    """Read the excel file and process it to get the different dataframes, including emissions from Z and Y."""
    emissions_Z = pd.read_excel(file_path_emissions_Z, index_col=0, header=0)
    emissions_Z = emissions_Z.iloc[:,1:]
    emissions_Z = emissions_Z[['GHG_total_OECD_consistent_ktco2eq', 'share_energy_related']].rename(columns={'GHG_total_OECD_consistent_ktco2eq': 'total_sectors'})
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
    emissions_total = emissions_total.assign(share_emissions_total_sectors = emissions_total['total_sectors'].div(emissions_total[['total_sectors', 'final_demand']].sum(axis=1).groupby(level='Country').sum()))  # we estimate the share of domestic emissions coming from different sectors
    emissions_total = emissions_total.assign(share_emissions_total_finaldemand=emissions_total['final_demand'].div(emissions_total[['total_sectors', 'final_demand']].sum(axis=1).groupby(level='Country').sum()))

    # share_emissions = emissions_total.div(emissions_total.sum(axis=1).groupby(level='Country').sum(), axis=0)

    return emissions_total

def get_main_stats(df, final_use):
    logging.info("Main statistics")
    sectors = df.sum(axis=1).to_frame(name='pmXi')
    sectors['pyi'] = df.sum(axis=0) + final_use.sum(axis=1)  # definition of total output, obtained from total intermediate consumption and final use

    Gamma = df.div(sectors['pyi'], axis=0)  # direct coefficient matrix
    Leontieff = np.linalg.inv(np.eye(len(Gamma)) - Gamma)  # Leontieff matrix
    Leontieff = pd.DataFrame(Leontieff, index=Gamma.index, columns=Gamma.columns)

    original_order = df.columns.get_level_values('Sector').unique()
    Omega = df.groupby(level='Sector', axis=1).sum()  # share of costs for each sector
    Omega = Omega[original_order]
    Omega = Omega.div(sectors['pmXi'], axis=0)

    Omega_energy = df.loc[:,df.columns.get_level_values('Sector').isin(ENERGY_SECTORS)]  # share of energy sectors in energy nest
    Omega_energy = Omega_energy.groupby(level='Sector', axis=1).sum().div(Omega_energy.sum(axis=1), axis=0)

    Omega_non_energy = df.loc[:,~df.columns.get_level_values('Sector').isin(ENERGY_SECTORS)]  # share of non-energy sector in non-energy nest
    Omega_non_energy = Omega_non_energy.groupby(level='Sector', axis=1).sum().div(Omega_non_energy.sum(axis=1), axis=0)

    # costs_energy = df.loc[:,df.columns.get_level_values('Sector').isin(ENERGY_SECTORS)].sum(axis=1).div(sectors['pmXi'], axis=0)  # share of costs going to energy sectors for each sector over total costs
    # costs_non_energy = 1 - costs_energy
    # costs_energy = pd.concat([costs_energy]*len(ENERGY_SECTORS), axis=1)
    # costs_energy.columns = ENERGY_SECTORS
    # number_non_energy_sectors = len(Gamma.index.get_level_values('Sector').unique()) - len(ENERGY_SECTORS)
    # costs_non_energy = pd.concat([costs_non_energy]*number_non_energy_sectors, axis=1)
    # costs_non_energy.columns = [x for x in Gamma.index.get_level_values('Sector').unique() if x not in ENERGY_SECTORS]
    # costs_energy = pd.concat([costs_energy, costs_non_energy], axis=1)
    # costs_energy.columns.name = 'Sector'

    costs_energy = df.loc[:,df.columns.get_level_values('Sector').isin(ENERGY_SECTORS)].sum(axis=1).div(sectors['pmXi'], axis=0)
    costs_energy = costs_energy.to_frame(name='Energy')
    costs_energy['Non-Energy'] = 1 - costs_energy['Energy']

    # share of intermediate consumption in dirty energy sectors
    sectors_dirty_energy = df.loc[:, df.columns.get_level_values('Sector').isin(DIRTY_ENERGY_SECTORS)]
    sectors_dirty_energy = sectors_dirty_energy.div(sectors_dirty_energy.sum(axis=1), axis=0)

    Domestic = df.div(df.groupby(level='Sector', axis=1).sum(), axis=0)  # share of costs for each variety for each sector
    # This could also be written as Domestic = df / df.groupby(level='Sector', axis=1).sum()

    # assert that the sum of Domestic, grouped by Sector  is equal to 1
    assert np.allclose(Domestic.groupby(level='Sector', axis=1).sum(), 1)

    psi = final_use.groupby('Sector').sum()  # sum of all final use
    psi = psi / psi.sum(axis=0)  # share of sector in final demand
    psi = psi.loc[original_order]
    psi_energy = final_use.loc[final_use.index.get_level_values('Sector').isin(ENERGY_SECTORS),:]  # share of energy sector in energy nest
    psi_energy = psi_energy.groupby(level='Sector', axis=0).sum().div(psi_energy.sum(axis=0))
    psi_non_energy = final_use.loc[~final_use.index.get_level_values('Sector').isin(ENERGY_SECTORS),:]  # share of non-energy sector in non-energy nest
    psi_non_energy = psi_non_energy.groupby(level='Sector', axis=0).sum().div(psi_non_energy.sum(axis=0))

    # costs_energy_final = final_use.loc[final_use.index.get_level_values('Sector').isin(ENERGY_SECTORS),:].sum(axis=0) / final_use.sum(axis=0)  # share of final demand going to energy sectors
    # costs_non_energy_final = 1 - costs_energy_final
    # costs_energy_final = pd.concat([costs_energy_final.to_frame().T]*len(ENERGY_SECTORS), axis=0)
    # costs_energy_final.index = ENERGY_SECTORS
    # costs_non_energy_final = pd.concat([costs_non_energy_final.to_frame().T]*number_non_energy_sectors, axis=0)
    # costs_non_energy_final.index = [x for x in Gamma.index.get_level_values('Sector').unique() if x not in ENERGY_SECTORS]
    # costs_energy_final = pd.concat([costs_energy_final, costs_non_energy_final], axis=0)
    # costs_energy_final.index.name = 'Sector'

    costs_energy_final = final_use.loc[final_use.index.get_level_values('Sector').isin(ENERGY_SECTORS),:].sum(axis=0) / final_use.sum(axis=0)
    costs_energy_final = costs_energy_final.to_frame(name='Energy')
    costs_energy_final['Non-Energy'] = 1 - costs_energy_final['Energy']
    costs_energy_final = costs_energy_final.T  # we transpose to have the same shape as psi


    psi_durable = final_use.loc[final_use.index.get_level_values('Sector').isin(DURABLE_GOODS),:]  # share of durable sectors in durable nest
    psi_durable = psi_durable.groupby(level='Sector', axis=0).sum().div(psi_durable.sum(axis=0))

    psi_non_durable = final_use.loc[final_use.index.get_level_values('Sector').isin(NON_DURABLE_GOODS),:]  # share of non durable sectors in non durable nest
    psi_non_durable = psi_non_durable.groupby(level='Sector', axis=0).sum().div(psi_non_durable.sum(axis=0))

    costs_durable_final = final_use.loc[final_use.index.get_level_values('Sector').isin(NON_DURABLE_GOODS),:].sum(axis=0) / final_use.sum(axis=0)
    costs_durable_final = costs_durable_final.to_frame(name='non_durable')
    costs_durable_final['energy_services'] = 1 - costs_durable_final['non_durable']
    costs_durable_final = costs_durable_final.T  # we transpose to have the same shape as psi

    xsi = final_use.div(final_use.groupby('Sector').sum())  # share of each variety in the nest for final demand

    # share of final consumption in dirty energy sectors, for each of the countries
    final_use_dirty_energy = final_use.loc[final_use.index.get_level_values('Sector').isin(DIRTY_ENERGY_SECTORS), :]
    final_use_dirty_energy = final_use_dirty_energy / final_use_dirty_energy.sum(axis=0)

    sectors['eta'] = 1 - sectors['pmXi'] / sectors['pyi']  # share of added value in total output

    sectors['phi'] = final_use.sum(axis=1) / sectors['pyi']
    for country in final_use.columns:
        sectors[f'phi_{country}'] = final_use[country] / sectors['pyi']

    # assert Leontieff.mul(phi.mul(pyi, axis=0)[col], axis=0).sum(axis=0)

    sectors['va'] = sectors['pyi'] - sectors['pmXi']  # we redefine va from other values, to ensure that data is correctly balanced

    sectors['gamma'] = value_added.loc['wli_over_vhi']

    Delta = df.div(sectors['pyi'], axis=1)  # Delta_ij =  expenditure on j by i as a share of total production of j

    empcomp = value_added.loc['EmpComp'].to_frame(name='share_labor')
    share_labor = empcomp.div(empcomp.groupby('Country').sum())
    share_labor_tot = (empcomp / empcomp.sum()).rename(columns={'share_labor': 'share_labor_tot'})
    # empcomp.squeeze() / empcomp.groupby('Country').sum().squeeze()   # other way to write that

    value_added.loc['Capital'] = value_added.loc['ValueAdded'] - value_added.loc['EmpComp']  # we define capital as the rest of the added value apart from labor
    assert (value_added.loc['Capital'] >= 0).all()
    capital = value_added.loc['Capital'].to_frame(name='share_capital')
    share_capital = capital.div(capital.groupby('Country').sum())  # we work here under the assumption that the wage is the same across all sectors, so that the share of labor is equal to the share of labor revenue
    share_capital_tot = (capital / capital.sum()).rename(columns={'share_capital': 'share_capital_tot'})

    sectors = pd.concat([sectors, share_labor, share_capital, share_labor_tot, share_capital_tot], axis=1)

    tmp = (sectors['gamma'] * sectors['va']).to_frame()  # payments for labor
    sectors['rev_labor'] =  tmp.div(sectors['va'].groupby('Country').sum(), axis=0).rename(columns={0: 'rev_labor'})  # share of total added value from country from labor revenues

    tmp = ((1-sectors['gamma']) * sectors['va']).to_frame()  # payments for capital
    sectors['rev_capital'] =  tmp.div(sectors['va'].groupby('Country').sum(), axis=0).rename(columns={0: 'rev_capital'})  # share of total added value from country from capital revenues

    total_GDP = sectors['va'].sum()
    assert total_GDP == sectors['va'].sum(), "Accounting equation is not verified for world GPD. It should be the same, whether calculated from value added or final use"
    share_GNE = sectors['va'].groupby('Country').sum() / total_GDP  # we rely on final_use to determine the share of each GNE in the total GDP (we could also have used value added)

    domestic_domar_weights = sectors['pyi'] / sectors['va'].groupby('Country').sum()
    sectors['domestic_domar_weights'] = domestic_domar_weights
    return sectors, Gamma, Leontieff, Omega, costs_energy, Omega_energy, Omega_non_energy, Domestic, psi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, xsi, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy


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

    sectors, Gamma, Leontieff, Omega, costs_energy, Omega_energy, Omega_non_energy, Domestic, psi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, xsi, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy = get_main_stats(df, final_use)
    calib = CalibOutput(sectors, emissions_total, xsi, psi, costs_energy_final, psi_energy, psi_non_energy, costs_durable_final, psi_durable, psi_non_durable, Omega, costs_energy, Omega_energy, Omega_non_energy, Gamma, Leontieff, Domestic, Delta, share_GNE, sectors_dirty_energy, final_use_dirty_energy, descriptions)
    calib.to_excel(f"outputs/calib_{country}.xlsx")
    calib2 = CalibOutput.from_excel(f"outputs/calib_{country}.xlsx")
    assert calib.equals(calib2)

# networks_stats(Gamma, col_final_use, total_output)