import numpy as np
import pandas as pd
from typing import Sequence
from matplotlib import pyplot as plt
import datetime
from pathlib import Path
from matplotlib.ticker import MaxNLocator


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


def parse_outputs(folder, list_sectors, post_processing, reference=None):
    """List files from a given folder according to conditions, and create a dictionary containing the files and corresponding configuration names. The goal is to automate the process.
    folder: Path
        Path to the folder containing the files
    list_sectors: list
        List of sectors to be considered
    post_processing: str
        Type of post-processing to be considered. It can be 'reference', 'sensitivity_energyservices' or 'sensitivity_energydurable'.
        If 'reference', the function will look for the file with the reference configuration.
        If 'sensitivity_energyservices', the function will look for the files with the reference configuration and different values of kappa.
    reference: dict
        Reference configuration. Default is None. If None, the reference configuration is {'theta': 0.5, 'sigma': 0.9, 'epsilon': 0.001, 'delta': 0.9, 'mu': 0.9, 'nu': 0.001, 'kappa': 0.5, 'rho': 0.95}
    """
    assert post_processing in ['reference', 'sensitivity_energyservices', 'sensitivity_durable']
    if reference is None:
        reference = {'theta': 0.5, 'sigma': 0.9, 'epsilon': 0.001, 'delta': 0.9, 'mu': 0.9, 'nu': 0.001, 'kappa': 0.5, 'rho': 0.95}
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
            elif post_processing == 'sensitivity_energyservices':
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

            elif post_processing == 'sensitivity_durable':
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

    save_fig(fig, save=save_with_date)


def absolute_emissions_barplot(df, sector, save=None, colors=None, figsize=(7, 7), fontsize_annotation=12, labelpad=30,
                               fontsize_big_xlabel=15, fontsize_small_xlabel=13, label_y_offset=20, annot_offset_neg=-15,
                               rotation=0, y_max=None, y_min=None):
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
        save = Path(save + f"_{sector}_{d}.pdf")
    save_fig(fig, save=save)