import numpy as np
import pandas as pd
from typing import Sequence


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