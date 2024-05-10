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

import datetime
import pandas as pd
from utils import EquilibriumOutput
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy as np

"""
This module includes all the functions used to parse the outputs of the model and create the plots.
"""

COLORS_DICT = {
    "Power": "#377EB8",  # Deep Sky Blue
    "Pig": "#FF7F00",  # Vivid Tangerine
    "Cattle": "#4DAF4A",  # Grass Green
    "FruitNut": "#F781BF",  # Soft Pink
    "BeefMeat": "#A65628",  # Saddle Brown
    "CerOth": "#984EA3",  # Rich Lilac
    "FoodOth": "#999999",  # Light Gray
    "AnimOth": "#E41A1C",  # Crimson
    "Road": "#D2B55B",  # Autumn Gold
    "IronSteel": "#2B9AC8",  # Cerulean Blue
    "ConstCivil": "#008080",  # Teal
    "FuelDist": "#333333",  # Dark Slate Gray
    "Petro": "#D95F02",  # Pumpkin Orange
    "FinalDemand": "#56B4E9",  # Jade Green
    "Legume": "#808000",  # Olive Green
    "others_negative": "#A6CEE3",  # Powder Blue
    "others_positive": "#E69F00",  # Burnt Sienna
    "WaterTrans": "#F0E442",  # Pale Lavender
    "Air": "#FF00FF",  # Magenta
    "Veget": "#FFFF33",  # Lemon Yellow
    "VegetProd": "#4DAF4A",  # Lemon Yellow
    "Gas": "#A65628",  # Lemon Yellow
    "Ceramics": "#008080",  # Lemon Yellow
    "FishProd": "#984EA3",
    "Coal": "#FFFF33"
}


TRANSFORM_LEGEND = {
    "Cattle": "Raising Cattle",
    "Pig": "Raising Pig",
    "FruitNut": "Growing Fruits",
    "Power": "Power",
    "BeefMeat": "Beef Meat",
    "FoodPth": "Food Products",
    "AnimOth": "Raising of Animals",
    "Road": "Road Transport",
    "Petro": "Petroleum Products",
    "IronSteel": "Iron and Steel",
    "FuelDist": "Gas Distribution",
    "ConstCivil": "Civil Construction",
    "FinalDemand": "Households Fossil Fuels Consumption",
    "Legume": "Growing Leguminous",
    "others_negative": "Other Sectors",
    "others_positive": "Other Sectors",
    "WaterTrans": "Water Transport",
    "Air": "Air Transport",
    "Veget": "Growing Vegetables",
    "VegetProd": "Vegetable Products",
    "CerOth": "Growing Cereals",
    "FoodOth": "Food Products",
    "Gas": "Gas extraction",
    "food": "Food shock",
    "energy": "Energy shock",
    "IO": "Input-Output",
    "D": "Demand",
    "D+CD": "Cobb-Douglas",
    "D+CD+CES": "CES"
}

def parse_outputs(folder, list_sectors, post_processing, file_selection='reference', configref=None):
    """List files from a given folder according to conditions, and create a dictionary containing the files and corresponding configuration names. The goal is to automate the process.
    folder: Path
        Path to the folder containing the files
    list_sectors: list
        List of sectors to be considered
    post_processing: str
        Type of post-processing to be considered. It can be 'reference', 'sensitivity_energyservices' or 'sensitivity_energydurable'.
        If 'reference', the function will look for the file with the reference configuration.
        If 'sensitivity_energyservices', the function will look for the files with the reference configuration and different values of kappa.
    configref: dict
        Reference configuration. Default is None. If None, the reference configuration is {'theta': 0.5, 'sigma': 0.9, 'epsilon': 0.001, 'delta': 0.9, 'mu': 0.9, 'nu': 0.001, 'kappa': 0.5, 'rho': 0.95}
    """
    assert post_processing in ['reference', 'sensitivity_energyservices', 'sensitivity_durable', 'sensitivity_production']
    reference = {'theta': 0.5, 'sigma': 0.9, 'epsilon': 0.001, 'delta': 0.9, 'mu': 0.9, 'nu': 0.001, 'kappa': 0.5, 'rho': 0.9}
    if configref is not None:
        assert isinstance(configref, dict)
        reference.update(configref)

    d = {}
    # list files inside folder
    for path in folder.iterdir():
        if path.is_file() and path.suffix == '.xlsx':
            file_name = path.stem
            if file_selection == 'heterogeneous' and '_heterogeneous' not in file_name:
                continue
            elif file_selection == 'uniform' and '_uniform' not in file_name:
                continue
            elif file_selection is 'reference' and ('_heterogeneous' in file_name or '_uniform' in file_name):
                continue
            config = path.name.split('.xlsx')[0].split('_')
            sector = config[1]
            current_values = {}
            allowed_keys = ['theta', 'sigma', 'epsilon', 'delta', 'mu', 'nu', 'kappa', 'rho']
            share = None
            if file_selection != 'uniform':
                params = config[2:]
            else:
                params = config[2:-1]  # we remove the uniform specification in the end
            for param in params:  # get configuration from file name
                key = ''.join([char for char in param if char.isalpha()])
                value = ''.join([char for char in param if char.isdigit() or char == '.'])
                if key in allowed_keys:
                    current_values[key] = float(value)
                elif 'heterogeneous' in key:
                    share = value

            if file_selection == 'heterogeneous':
                if sector in list_sectors:
                    kappa_value = path.stem.split('kappa')[1].split('_')[0]
                    d[f'{sector} - {share} - {kappa_value}'] = path
            else:
                if post_processing == 'reference':
                    if all(reference[key] == current_values[key] for key in reference):
                        if sector in list_sectors:
                            d[sector] = path
                elif post_processing == 'sensitivity_energyservices':  # parameter between energy and durables
                    # Check if all parameters except kappa match their reference values
                    if all(reference[key] == current_values[key] for key in reference if key != 'kappa'):
                        kappa_name = 'Low' if current_values['kappa'] < 0.2 else 'Ref' if current_values[
                                                                                              'kappa'] < 0.9 else 'High'
                        if sector in list_sectors:
                            d[f'{sector} - {kappa_name}'] = path

                elif post_processing == 'sensitivity_durable':  # parameter between energy services and nondurables
                    # Check if all parameters except rho match their reference values
                    if all(reference[key] == current_values[key] for key in reference if key != 'rho'):
                        rho_name = 'Very Low' if current_values['rho'] < 0.2 else 'Low' if current_values[
                                                                                              'rho'] < 0.9 else 'Ref'
                        if sector in list_sectors:
                            d[f'{sector} - {rho_name}'] = path

                elif post_processing == 'sensitivity_production':  # parameter between energy services and nondurables
                    # Check if all parameters except nu match their reference values
                    if all(reference[key] == current_values[key] for key in reference if key != 'nu'):
                        nu_name = 'Ref' if current_values['nu'] < 0.2 else 'High' if current_values[
                                                                                              'nu'] < 0.9 else 'Very High'
                        if sector in list_sectors:
                            d[f'{sector} - {nu_name}'] = path

    return d


def post_processing_new_consumer(d, param='kappa'):
    """Add the value of param to the key of the dictionary. The function is used for plots only."""
    new_d = {}
    for key in d.keys():
        value = d[key].stem.split(param)[1].split('_')[0]
        new_d[f'{key} - {value}'] = d[key]
    return new_d



def parse_emissions(dict_paths, index_names, folderpath):
    """Creates an output file which contains the variation of emissions for each country and each sector, in correct format.
    Function is used for plots only."""
    # Get emissions variation
    concatenated_dfs = []
    concatenated_dfs_absolute = []
    concatenated_dfs_detail = []

    def rename_index(L, country):
        """Renames domestic country to 'Dom.' and the rest to 'RoW'"""
        new_index = []
        for i in L:
            tmp = i.split('_absolute')[0]
            if tmp == country:
                new_index.append('Dom.')
            else:
                new_index.append('RoW')
        return new_index

    for key in dict_paths.keys():
        equilibrium_output = EquilibriumOutput.from_excel(dict_paths[key])
        df = equilibrium_output.emissions_hat
        country = df.index[0]
        index_values = key.split(' - ')
        transformed_df = df.loc[country] - 1  # we transform relative variation to absolute variation
        transformed_df = transformed_df.to_frame().T
        transformed_df.index = pd.MultiIndex.from_tuples([tuple([country] + index_values)], names=['Aggregation'] + index_names)
        concatenated_dfs.append(transformed_df)

        emissions_absolute = df.loc[df.index.str.contains('_absolute')]

        emissions_absolute.index = rename_index(emissions_absolute.index, country)
        emissions_absolute.index = pd.MultiIndex.from_tuples([tuple([country] + index_values) + (idx,) for idx in emissions_absolute.index], names=['Aggregation'] + index_names + ['Country'])
        # emissions_absolute.index = pd.MultiIndex.from_product([tuple([country] + index_values), emissions_absolute.index], names=['Aggregation'] + index_names + ['Country'])
        concatenated_dfs_absolute.append(emissions_absolute)

        emissions_detail = equilibrium_output.emissions_detail
        emissions_detail.columns = pd.MultiIndex.from_tuples([(idx, index_values[0]) for idx in emissions_detail.columns], names=['Country', 'Sector'])
        emissions_detail.index = pd.MultiIndex.from_tuples([tuple(idx.split('-')) for idx in emissions_detail.index], names=['Sector', 'Category'])
        concatenated_dfs_detail.append(emissions_detail)
    emissions_df = pd.concat(concatenated_dfs, axis=0)
    if 'emissions_single' in emissions_df.columns:
        emissions_df = emissions_df.rename(columns={
            'emissions_IO': 'IO',
            'emissions_CD': 'D+CD',
            'emissions_ref': 'D+CD+CES',
            'emissions_single': 'D'
        })
        emissions_df = emissions_df.reindex(['IO', 'D', 'D+CD', 'D+CD+CES'], axis=1)
    else:
        emissions_df = emissions_df.rename(columns={
            'emissions_IO': 'IO',
            'emissions_CD': 'CD',
            'emissions_ref': 'CD+CES'
        })
        emissions_df = emissions_df.reindex(['IO', 'CD', 'CD+CES'], axis=1)
    emissions_df.columns.names = ['Effect']
    emissions_df = emissions_df.stack()

    emissions_absolute_df = pd.concat(concatenated_dfs_absolute, axis=0)
    if 'emissions_single' in emissions_absolute_df.columns:
        emissions_absolute_df = emissions_absolute_df.rename(columns={
            'emissions_IO': 'IO',
            'emissions_CD': 'D+CD',
            'emissions_ref': 'D+CD+CES',
            'emissions_single': 'D'
        })
        emissions_absolute_df = emissions_absolute_df.reindex(['IO', 'D', 'D+CD', 'D+CD+CES'], axis=1)
    else:
        emissions_absolute_df = emissions_absolute_df.rename(columns={
            'emissions_IO': 'IO',
            'emissions_CD': 'CD',
            'emissions_ref': 'CD+CES'
        })
        emissions_absolute_df = emissions_absolute_df.reindex(['IO', 'CD', 'CD+CES'], axis=1)
    emissions_absolute_df = emissions_absolute_df.stack()
    # rename last level of index from None to Effect
    # the level is the last level
    emissions_absolute_df.index = emissions_absolute_df.index.set_names('Effect', level=-1)

    emissions_detail_df = pd.concat(concatenated_dfs_detail, axis=1)

    if not (folderpath / Path('postprocess')).is_dir():
        os.mkdir(folderpath / Path('postprocess'))
    with pd.ExcelWriter(folderpath / Path('postprocess') / Path('emissions.xlsx')) as writer:
        emissions_df.to_excel(
            writer,
            header=True,
            index=True)

    with pd.ExcelWriter(folderpath / Path('postprocess') / Path('emissions_absolute_df.xlsx')) as writer:
        emissions_absolute_df.to_excel(
            writer,
            header=True,
            index=True)
    return emissions_df, emissions_absolute_df, emissions_detail_df

def plot_domestic_emissions(series, country, folder_path, ratio_space=0.1, y_text_offset=0.15, y_label_categories = 0.05,
                   y_label_sector = 0.60, textwidth_heith=200, textwidth_width=202):
    """
    Function where all the space values in the graph are calculated in the function.
    --- Parameters ---
    ratio_space: float
        Size of blank space over size of ensemble of categories
    y_text_offset: float
        Offset of the text below the bar
    y_label_categories: float
        Offset of the label of the category
    y_label_sector: float
        Offset of the label of the sector
    textwidth_width: int
        Get the textwidth from latex with \the\textwidth (or \showthe\textwidth)
    textwidth_heith: int
        Get the textwidth from latex with \the\textwidth (or \showthe\textwidth)
    """

    n_countries = series.index.get_level_values("Country").nunique()
    n_sectors = series.index.get_level_values("Sector").nunique()
    n_categories = series.index.get_level_values("Category").nunique()
    min_value = series.min()

    # Some sizes in the plot, see figure for details

    # Half-Blank Sector1 Blank Sector2 ... SectorN Half-Blank
    size_sectors = 1 / (n_sectors + ratio_space * n_sectors)
    size_blank = ratio_space * size_sectors

    # For each sector, multiple categories displayed
    size_categories = size_sectors / n_categories

    ## Select the country
    s_country = series.xs(country, level='Country')

    ## The ax: we plot a single black line from 0 to 1

    # Get the textwidth from latex with \the\textwidth (or \showthe\textwidth), so that the plot is done exactly for the available space.
    fig, ax = plt.subplots(figsize=(textwidth_width / 72, textwidth_heith / 72))
    ax.spines['left'].set_visible(False)  # hide the left spine
    ax.spines['right'].set_visible(False)  # hide the right spine
    ax.spines['bottom'].set_visible(False)  # hide the top spine
    ax.spines['top'].set_visible(False)  # hide the top spine
    ax.yaxis.set_visible(False)  # hide the y-axis
    ax.xaxis.set_visible(False)  # hide the y-axis

    # Some tricks in order to avoid the axis to change position between the two countries
    ax.plot([0, 1], [0, 0], color='black')
    ax.plot([0, 0], [0, min_value], color="white", alpha=0)
    plt.ylim(min_value, 0.5)
    plt.xlim(0, 1)
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.1)

    ## Select the palette

    # A color-blind friendly palette
    palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    # Default palette
    # palette = cm.tab10(9)

    colors = {
        k: v for k, v in zip(
            s_country.index.get_level_values("Category").unique(),
            palette)}

    # Some keyword so that s in ax.text(x, y, s) is aligned around (x, y)
    text_align = {
        "horizontalalignment": "center", "verticalalignment": "center"
    }

    # We start with a blank offset
    current_x = size_blank / 2

    # For each first_level value...
    for ind_sector, sector in enumerate(s_country.index.get_level_values('Sector').unique()):
        # Restrict the dataframe
        s_sector = s_country.xs(sector, level="Sector")
        # Print the name of the sector:
        # Leave half a blank, then multiple sectors, then multiple full blanks
        ax.text(size_blank * (ind_sector + 0.5) + size_sectors * (ind_sector + 0.5), y_label_sector, sector,
                **text_align, fontsize=8)

        # For each second_level value...
        for category in s_sector.index.get_level_values('Category').unique():
            # This is the value we plot
            current_y = s_sector.loc[category]
            plt.bar(current_x, current_y, width=size_categories, color=colors[category], align="edge")

            # This is the label
            ax.text(current_x + size_categories / 2, y_label_categories, category,
                    horizontalalignment="center", verticalalignment="bottom", rotation=45, fontsize=7)
            # This is the value, below the bar
            ax.text(current_x + size_categories / 2, current_y - y_text_offset, f"{current_y:.2f}%",
                    **text_align, fontsize=6, rotation=0)
            # Next plot will be a bit further
            current_x += size_categories

        # We skip a blank
        current_x += size_blank

    # fig.savefig(f"output_{country}.pdf", transparent=True, bbox_inches=Bbox([[0, min_value], [1, 0.5]]), pad_inches=0)
    d = datetime.datetime.now().strftime("%m%d%H%M%S")
    fig.savefig(folder_path / Path(f"domestic_emissions_{country}_{d}.pdf"), transparent=True, bbox_inches=None, pad_inches=0)



def format_ax_new(ax, y_label=None, title=None, format_x=None,
                  format_y=lambda y, _: y, ymin=None, ymax=None, xinteger=True, xmin=None, x_max=None, loc_title=None,
                  c_title=None):
    """

    Parameters
    ----------
    y_label: str
    format_y: function
    ymin: float or None
    xinteger: bool
    title: str, optional

    Returns
    -------

    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_linewidth(2)

    ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if format_x is not None:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        if loc_title is not None:
            ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    if xmin is not None:
        ax.set_xlim(xmin=xmin)
        _, x_max = ax.get_xlim()
        ax.set_xlim(xmax=x_max * 1.1)

    if ymin is not None:
        ax.set_ylim(ymin=0)
        _, y_max = ax.get_ylim()
        ax.set_ylim(ymax=y_max * 1.1)

    if ymax is not None:
        ax.set_ylim(ymax=ymax, ymin=ymin)

    if xinteger:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax


def save_fig(fig, save=None, bbox_inches='tight'):
    if save is not None:
        fig.savefig(save, bbox_inches=bbox_inches)
        plt.close(fig)
    else:
        plt.show()


# Function to create the plot
def domestic_emissions_barplot(df, save=None, colors=None, figsize=(7, 7), fontsize_annotation=12, labelpad=30,
                               fontsize_big_xlabel=15, fontsize_small_xlabel=13, annot_offset_neg=-15, rotation=0, first_level_index='Sector',
                               second_level_index='Effect', filesuffix = '.pdf', dict_legend=None):
    """
    Plot emissions by sector and category
    --- Parameters ---
    df: pd.DataFrame
        DataFrame with a MultiIndex with levels "Sector" and "Category"
    save: str
        Path to save the plot
    colors: dict
        Dictionary with the color for each category
    figsize: tuple
        Size of the figure
    fontsize_annotation: int
        Font size of the annotations
    labelpad: int
        Space between the xlabel and the plot
    fontsize_label: int
        Font size of the xlabel
    rotation: int
        Rotation of the xticks
    """
    if colors is None:
        palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        colors = {k: v for k, v in zip(df.index.get_level_values(second_level_index).unique(), palette)}
    list_keys = df.index.get_level_values(first_level_index).unique()

    # Setup the figure and axes
    fig, axes = plt.subplots(1, int(len(list_keys)), figsize=figsize, sharex='all', sharey='all')
    if len(list_keys) == 1:
        axes = [axes]  # Make it iterable

    # Plotting
    for ax, sector in zip(axes, list_keys):
        sector_data = df.xs(sector, level=first_level_index)
        bars = ax.bar(sector_data.index, sector_data.values,
                      color=[colors.get(x, '#333333') for x in sector_data.index], width=1)

        # ax = format_ax_new(ax)
        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.yaxis.set_tick_params(which=u'both', length=0)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_small_xlabel, rotation=rotation)

        # Retrieve current tick positions and labels for customization
        tick_positions = ax.get_xticks()
        tick_labels = [dict_legend.get(label.get_text(), label.get_text()) if dict_legend is not None else label for label in ax.get_xticklabels()]

        # Set the custom tick positions and labels
        ax.set_xticks(tick_positions)  # This fixes the ticks to their current positions
        ax.set_xticklabels(tick_labels)  # Apply the new labels

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            annotation_offset = 6 if height > 0 else annot_offset_neg
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, annotation_offset),  # Move text above/below bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=fontsize_annotation)

        # X-axis labels on top
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=1, prune='upper'))
        ax.axhline(y=0, color='black', linewidth=2)
        sector_name = sector if dict_legend is None else (dict_legend[sector] if sector in dict_legend.keys() else sector)
        ax.set_xlabel(sector_name, labelpad=labelpad,
                      fontsize=fontsize_big_xlabel)  # labelpad is the space between the xlabel and the plot.

        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        ax.yaxis.set_ticks([])

    # Adjust the layout
    plt.tight_layout()

    if save is not None:
        d = datetime.datetime.now().strftime("%m%d")

        # Ensure save is a Path object
        if not isinstance(save, Path):
            save = Path(save)

        # Append the date to the filename before the extension
        save_with_date = save.parent / (save.stem + f"_{d}" + filesuffix)
    else:
        save_with_date = None

    save_fig(fig, save=save_with_date)


def absolute_emissions_barplot(df, sector, save=None, colors=None, figsize=(7, 7), fontsize_annotation=12, labelpad=30,
                               fontsize_big_xlabel=15, fontsize_small_xlabel=13, label_y_offset=20, annot_offset_neg=-15,
                               rotation=0, y_max=None, y_min=None, dict_legend=None, filesuffix='.pdf'):
    """
    Plot emissions by sector and category
    --- Parameters ---
    df: pd.DataFrame
        DataFrame with a MultiIndex with levels "Sector" and "Category"
    save: str
        Path to save the plot
    colors: dict
        Dictionary with the color for each category
    figsize: tuple
        Size of the figure
    fontsize_annotation: int
        Font size of the annotations
    labelpad: int
        Space between the xlabel and the plot
    fontsize_label: int
        Font size of the xlabel
    rotation: int
        Rotation of the xticks
    """
    if colors is None:
        palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        colors = {k: v for k, v in zip(df.index.get_level_values("Country").unique(), palette)}
    list_keys = df.index.get_level_values('Effect').unique()

    # Setup the figure and axes
    fig, axes = plt.subplots(1, int(len(list_keys)), figsize=figsize, sharex='all', sharey='all')
    if len(list_keys) == 1:
        axes = [axes]  # Make it iterable

    # Plotting
    for ax, effect in zip(axes, list_keys):
        sector_data = df.xs(effect, level='Effect')
        bars = ax.bar(sector_data.index, sector_data.values,
                      color=[colors.get(x, '#333333') for x in sector_data.index], width=1)

        # ax = format_ax_new(ax)
        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.yaxis.set_tick_params(which=u'both', length=0)
        #  ax.tick_params(axis='x', which='major', labelsize=12, rotation=rotation)
        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            annotation_offset = 6 if height > 0 else annot_offset_neg
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, annotation_offset),  # Move text above/below bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=fontsize_annotation)

        # X-axis labels on top
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=1, prune='upper'))
        ax.axhline(y=0, color='black', linewidth=2)
        effect_name = effect if dict_legend is None else (dict_legend[effect] if effect in dict_legend.keys() else effect)
        ax.set_xlabel(effect_name, labelpad=labelpad,
                      fontsize=fontsize_big_xlabel)  # labelpad is the space between the xlabel and the plot.

        if y_max is not None:
            ax.set_ylim(ymax=y_max)
        if y_min is not None:
            ax.set_ylim(ymin=y_min)

        # ax.xaxis.set_label_position('top')
        # ax.xaxis.tick_top()

        ax.yaxis.set_ticks([])

        ax.set_xticklabels([])

        # We modify the position of the x-axis labels
        for i, bar in enumerate(bars):
            bar_value = sector_data.values[i]
            effect_name = sector_data.index[i]

            # Adjust label_y_offset based on the sign of the bar_value
            adjusted_label_y_offset = - label_y_offset * np.sign(bar_value)

            # Use ax.text to place each x-axis label individually
            ax.text(i, adjusted_label_y_offset, effect_name,
                    ha='center', va='bottom' if bar_value > 0 else 'top',
                    fontsize=fontsize_small_xlabel, rotation=rotation)


    # Adjust the layout
    plt.tight_layout()

    if save is not None:
        d = datetime.datetime.now().strftime("%m%d")

        # Ensure save is a Path object
        if not isinstance(save, Path):
            save = Path(save)

        # Append the date to the filename before the extension
        save_with_date = save.parent / (save.stem + f"_{sector}_{d}" + filesuffix)
    else:
        save_with_date = None
    save_fig(fig, save=save_with_date)


def emissions_breakdown_barplot(df, figsize=(7, 7), groupby='Type', format_y=lambda y, _: '{:.1f} %'.format(y), hline=True,
                                display_title=True, rotation=0, save=None, filesuffix='.pdf', legend_loc='right', fontsize_big_xlabel=15,
                                fontsize_small_xlabel=13, fontsize_legend_labels=13, dict_legend=None):
    """Displays the breakdown of emissions by sector and category"""
    n_columns = int(len(df.columns))
    y_max = df[df > 0].groupby([i for i in df.index.names if i != groupby]).sum().max().max() * 1.1
    y_min = df[df < 0].groupby([i for i in df.index.names if i != groupby]).sum().min().min() * 1.1

    top_categories_sets = [set(df[col].abs().nlargest(10).index) for col in df]
    all_top_categories = set.union(*top_categories_sets)

    others_positive_sum = df.apply(lambda col: col[(~col.index.isin(all_top_categories)) & (col > 0)].groupby('Category').sum())
    others_positive_sum.index = pd.MultiIndex.from_product([['others_positive'], others_positive_sum.index], names=['Sector', 'Category'])
    others_negative_sum = df.apply(lambda col: col[(~col.index.isin(all_top_categories)) & (col < 0)].groupby('Category').sum())
    others_negative_sum.index = pd.MultiIndex.from_product([['others_negative'], others_negative_sum.index], names=['Sector', 'Category'])


    # Remark: there may be a bug if others_positive_sum or others_negative_sum is empty
    df = pd.concat([df, others_positive_sum, others_negative_sum], axis=0)

    # add others_positive_sum index to all_top_categories automatically from others_positive_sum.index

    df = df.drop(index=df.index[~df.index.isin(all_top_categories.union(set(others_positive_sum.index).union(set(others_negative_sum.index))))])

    color_dict = {k: COLORS_DICT[k] if k in COLORS_DICT else "#333333" for k in df.index.get_level_values('Sector').unique()}

    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, sharex='all', sharey='all')
    handles, labels = None, None
    unique_handles = {}
    for k in range(n_rows * n_columns):
        column = k % n_columns
        if n_rows * n_columns == 1:  # in this case, we have a single plot
            ax = axes
        else:
            ax = axes[column]

        try:
            key = df.columns[k]
            df_temp = df[key].sort_values(ascending=True).unstack(groupby)

            column_colors = [color_dict.get(x, 'grey') for x in df_temp.columns]

            df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=column_colors)

            ax = format_ax_new(ax, format_y=format_y, xinteger=True)
            if (y_max > 0.6) or (y_min < -0.6):
                ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 0.5, 0.5))
            else:
                ax.set_yticks(np.arange(y_min, y_max + 0.02, 0.02))

            # ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
            ax.spines['left'].set_visible(False)
            ax.set_ylim(ymax=y_max)
            ax.set_ylim(ymin=y_min)
            ax.set_xlabel('')

            if hline:
                ax.axhline(y=0, c='black')

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.tick_params(axis='both', which='major', labelsize=fontsize_small_xlabel)

            title = key if dict_legend is None else (dict_legend[key] if key in dict_legend.keys() else key)
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            if display_title:
                ax.set_title(title, pad=-1.6, fontsize=fontsize_big_xlabel)
            else:
                ax.set_title('')

            current_handles, current_labels = ax.get_legend_handles_labels()
            for handle, label in zip(current_handles, current_labels):
                # Update the unique handles dictionary with the latest handle for each label
                unique_handles[label] = handle

            # Remove the legend from the individual subplot
            ax.get_legend().remove()

        except IndexError:
            ax.axis('off')

        labels, handles = zip(*list(unique_handles.items()))
        if dict_legend is not None:
            labels = [dict_legend.get(e, e) for e in labels]
            labels = [dict_legend[e] if e in dict_legend.keys() else e for e in labels]

        # Create the legend for the figure
        if legend_loc == 'lower':
            fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=fontsize_legend_labels)
        else:
            fig.legend(handles, labels, loc='center left', frameon=False, ncol=1, bbox_to_anchor=(1, 0.5), fontsize=fontsize_legend_labels)
        # Adjust the layout
        plt.tight_layout()

    if save is not None:
        d = datetime.datetime.now().strftime("%m%d")

        # Ensure save is a Path object
        if not isinstance(save, Path):
            save = Path(save)

        # Append the date to the filename before the extension
        save_with_date = save.parent / (save.stem + f"_{d}" + filesuffix)
    else:
        save_with_date = None
    save_fig(fig, save=save_with_date)

