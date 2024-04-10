import numpy as np
import pandas as pd
from typing import Sequence
from matplotlib import pyplot as plt
import datetime
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
import os


@dataclass
class EquilibriumOutput:
    """Class to save the outcome of the model."""
    pi_hat: pd.DataFrame
    yi_hat: pd.DataFrame
    pi_imports_finaldemand: pd.DataFrame
    final_demand: pd.DataFrame
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
                (self.pi_imports_finaldemand, "pi_imports_finaldemand"),
                (self.final_demand, "final_demand"),
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
                if "Sector" in current_df.index.names and sheet_name != "descriptions" and sheet_name != "emissions_detail":
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
            pi_imports_finaldemand = pd.read_excel(xls, sheet_name="pi_imports_finaldemand", index_col=0).drop(columns="long_description")
            final_demand = unflatten_index_in_df(
                pd.read_excel(xls, sheet_name="final_demand", index_col=0)
                .drop(columns="long_description"),
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
        return cls(pi_hat, yi_hat, pi_imports_finaldemand, final_demand, domar, labor_capital, emissions_hat, emissions_detail, global_variables, descriptions)

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
    return (
            df1.index.equals(df2.index)
            and df1.columns.equals(df2.columns)
            and df1.dtypes.equals(df2.dtypes)
            and np.allclose(df1.values, df2.values)
    )


def parse_outputs(folder, list_sectors, post_processing, configref=None):
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
    assert post_processing in ['reference', 'sensitivity_energyservices', 'sensitivity_durable']
    reference = {'theta': 0.5, 'sigma': 0.9, 'epsilon': 0.001, 'delta': 0.9, 'mu': 0.9, 'nu': 0.001, 'kappa': 0.5, 'rho': 0.95}
    if configref is not None:
        assert isinstance(configref, dict)
        reference.update(configref)
    d = {}
    # list files inside folder
    for path in folder.iterdir():
        if path.is_file() and path.suffix == '.xlsx':
            config = path.name.split('.xlsx')[0].split('_')
            domestic_country = config[0]
            sector = config[1]
            for key in config[2:]:  # we get the different parameters from the configuration
                if 'theta' in key:
                    theta = float(key.split('theta')[1])
                if 'sigma' in key:
                    sigma = float(key.split('sigma')[1])
                if 'epsilon' in key:
                    epsilon = float(key.split('epsilon')[1])
                if 'delta' in key:
                    delta = float(key.split('delta')[1])
                if 'mu' in key:
                    mu = float(key.split('mu')[1])
                if 'nu' in key:
                    nu = float(key.split('nu')[1])
                if 'kappa' in key:
                    kappa = float(key.split('kappa')[1])
                if 'rho' in key:
                    rho = float(key.split('rho')[1])
            current_values = {
                'theta': theta,
                'sigma': sigma,
                'epsilon': epsilon,
                'delta': delta,
                'mu': mu,
                'nu': nu,
                'kappa': kappa,
                'rho': rho
            }

            if post_processing == 'reference':
                if all(reference[key] == current_values[key] for key in reference):
                    if sector in list_sectors:
                        d[sector] = path
            elif post_processing == 'sensitivity_energyservices':  # parameter between energy and durables
                # Check if all parameters except kappa match their reference values
                if all(reference[key] == current_values[key] for key in reference if key != 'kappa'):
                    if kappa < 0.2:
                        kappa_name = 'Low'
                    elif kappa < 0.9:
                        kappa_name = 'Ref'
                    else:
                        kappa_name = 'High'
                    if sector in list_sectors:
                        d[f'{sector} - {kappa_name}'] = path

            elif post_processing == 'sensitivity_durable':  # parameter between energy services and nondurables
                # Check if all parameters except kappa match their reference values
                if all(reference[key] == current_values[key] for key in reference if key != 'rho'):
                    if rho < 0.2:
                        rho_name = 'Low'
                    elif rho < 0.9:
                        rho_name = 'Ref'
                    else:
                        rho_name = 'High'
                    if sector in list_sectors:
                        d[f'{sector} - {rho_name}'] = path
    return d


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
    emissions_df = emissions_df.rename(columns={
        'emissions_IO': 'IO',
        'emissions_CD': 'D+CD',
        'emissions_ref': 'D+CD+CES',
        'emissions_single': 'D'
    })
    emissions_df = emissions_df.reindex(['IO', 'D', 'D+CD', 'D+CD+CES'], axis=1)
    emissions_df.columns.names = ['Effect']
    emissions_df = emissions_df.stack()

    emissions_absolute_df = pd.concat(concatenated_dfs_absolute, axis=0)
    emissions_absolute_df = emissions_absolute_df.rename(columns={
        'emissions_IO': 'IO',
        'emissions_CD': 'D+CD',
        'emissions_ref': 'D+CD+CES',
        'emissions_single': 'D'
    })
    emissions_absolute_df = emissions_absolute_df.reindex(['IO', 'D', 'D+CD', 'D+CD+CES'], axis=1)
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
                               second_level_index='Effect', filesuffix = '.pdf'):
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
        ax.set_xlabel(sector, labelpad=labelpad,
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
                               rotation=0, y_max=None, y_min=None, filesuffix='.pdf'):
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
        ax.set_xlabel(effect, labelpad=labelpad,
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


def emissions_breakdown_barplot(df, figsize=(7, 7), groupby='Type', colors=None, format_y=lambda y, _: '{:.1f}'.format(y), hline=True,
                                display_title=True, rotation=0, save=None, filesuffix='.pdf', legend_loc='right', fontsize_big_xlabel=15, fontsize_small_xlabel=13):
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

    palette = [
    "#377EB8",  # Deep Sky Blue
    "#FF7F00",  # Vivid Tangerine
    "#4DAF4A",  # Grass Green
    "#F781BF",  # Soft Pink
    "#A65628",  # Saddle Brown
    "#984EA3",  # Rich Lilac
    "#999999",  # Light Gray
    "#E41A1C",  # Crimson
    "#DEDE00",  # Mustard
    "#2B9AC8",  # Cerulean Blue
    "#008080",  # Teal
    "#333333",  # Dark Slate Gray
    "#D95F02",  # Pumpkin Orange
    "#56B4E9",  # Jade Green
    "#808000",   # Olive Green
    "#A6CEE3",  # Powder Blue
    "#E69F00",  # Burnt Sienna
    "#F0E442",  # Pale Lavender
    "#FF00FF",  # Magenta
    "#FFFF33"  # Lemon Yellow
    ]


    # Remark: there may be a bug if others_positive_sum or others_negative_sum is empty
    df = pd.concat([df, others_positive_sum, others_negative_sum], axis=0)

    # add others_positive_sum index to all_top_categories automatically from others_positive_sum.index

    df = df.drop(index=df.index[~df.index.isin(all_top_categories.union(set(others_positive_sum.index).union(set(others_negative_sum.index))))])

    color_list = colors if colors is not None else palette
    color_dict = {k: v for k, v in zip(df.index.get_level_values('Sector').unique(), color_list)}

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

            # ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
            ax.spines['left'].set_visible(False)
            ax.set_ylim(ymax=y_max)
            ax.set_ylim(ymin=y_min)
            ax.set_xlabel('')

            if hline:
                ax.axhline(y=0)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.tick_params(axis='both', which='major', labelsize=fontsize_small_xlabel)

            title = key
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

        # Create the legend for the figure
        if legend_loc == 'lower':
            fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.1))
        else:
            fig.legend(handles, labels, loc='center left', frameon=False, ncol=1, bbox_to_anchor=(1, 0.5))
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

