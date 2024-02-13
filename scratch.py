import pandas as pd
from itertools import product
import numpy as np


index = pd.MultiIndex.from_tuples(
    product(("France", "US"), ("Food", "Transport", "Wheat")),
    names=["Country", "Sector"],
)
gamma = pd.DataFrame(
    index=index, columns=index, data=np.arange(len(index)**2).reshape(len(index), len(index))
)

p = pd.Series(index=gamma.index[::-1], data=np.arange(6))

col_final_use = pd.Series(index=index, data=np.random.rand(len(index)))
total_output = pd.Series(index=index, data=np.random.rand(len(index)))

df = pd.DataFrame(index=index, columns=['A'], data=np.random.rand(len(index)))
s = pd.Series(index=index, data=np.random.rand(len(index)))
s = s.reindex(s.index[::-1])
df['B'] = s

s2 = pd.Series(index=index, data=np.random.rand(len(index)))
s2 = s2.reindex(s2.index[::-1])
df = pd.concat([df, s2], axis=1)


s = pd.Series(index=index, data=np.arange(len(index)))
s.index.names = ["Country", "Sector"]
t = pd.DataFrame(index=["Food", "Transport", "Wheat"], columns = ["France", "US"], data=np.arange(6).reshape(3, 2))
t.index.names = ["Sector"]
t.columns.names = ["Country"]


s = pd.Series(index=index, data=np.arange(len(index)))
s.index.names = ["Country", "Sector"]
t = pd.Series(index=["France", "US"], data=np.arange(2))
t.index.names = ["Country"]