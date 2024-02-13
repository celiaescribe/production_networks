import pandas as pd
import numpy as np
import logging
from calibrate import CalibOutput
from scipy.optimize import fsolve, root
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def residuals(lvec, li_hat,ki_hat, betai_hat,theta,sigma,epsilon,delta,mu,sectors, xsi, psi, Omega, Domestic, Delta, share_GNE, singlefactor=False, domestic_country = 'FRA'):
    print('hey')
    N = len(sectors)  # number of sectors times countries
    S = len(psi)  # number of sectors
    C = xsi.shape[1] # number of countries
    vec = np.exp(lvec)
    # vec = lvec
    pi_hat = vec[:N]
    yi_hat = vec[N:2*N]
    PSigmaY = pd.DataFrame(columns=psi.columns, data=vec[2*N:2*N+C].reshape(1, -1))

    pi_hat = pd.Series(pi_hat, index=sectors.index)
    yi_hat = pd.Series(yi_hat, index=sectors.index)

    betai_hat = betai_hat.to_frame()
    betai_hat = betai_hat.rename(columns={betai_hat.columns[0]: domestic_country})
    betai_hat = betai_hat.reindex(psi.columns, axis=1, fill_value=1)

    if singlefactor:
        w_domestic = pd.DataFrame(columns=psi.columns, data=vec[2*N+C:2*N+2*C].reshape(1, -1))
        r_domestic = pd.DataFrame(columns=psi.columns, data=vec[2*N+2*C:2*N+3*C].reshape(1, -1))

    assert Domestic.index.equals(Domestic.columns), "Domestic index and columns are not the same while they should be"
    # We create a price dataframe with the same dimensions as the Domestic dataframe
    # With the following syntax, we assume that pi_hat follows the order of Domestic columns. Important !!
    # price_df = pd.concat([pd.DataFrame(columns=Domestic.columns, data=pi_hat.reshape(1, -1))] * len(Domestic.index), axis=0, ignore_index=True)
    # price_df.index = Domestic.index

    # Price for imports nest
    if delta == 1:
        price_imports = Domestic.mul(np.log(price_df))
        price_imports = np.exp(price_imports.groupby(level='Sector', axis=1).sum())
    else:
        price_imports = Domestic.mul(price_df**(1-delta))
        price_imports = (price_imports.groupby(level='Sector', axis=1).sum())**(1/(1-delta))
    price_imports = price_imports.reindex(Omega.columns, axis=1)
    assert price_imports.shape == Omega.shape

    # Price for intermediate sectors goods
    if epsilon == 1:
        price_intermediate = Omega.mul(np.log(price_imports))
        price_intermediate = np.exp(price_intermediate.sum(axis=1))
    else:
        price_intermediate = Omega.mul(price_imports**(1-epsilon))
        price_intermediate = price_intermediate.sum(axis=1)**(1/(1-epsilon))
    price_intermediate = price_intermediate.to_frame()
    assert price_intermediate.shape == (N,1)

    if singlefactor:
        pass
    else:
        hi_hat = li_hat**(sectors['gamma']) * ki_hat**(1-sectors['gamma'])
        vi_hat = pi_hat * (yi_hat / hi_hat)**(1/theta)
        wi_hat = vi_hat * hi_hat / li_hat
        ri_hat = vi_hat * hi_hat / ki_hat

    # Price for import nest, final demand
    price_df = pd.concat([pd.DataFrame(index=xsi.index, data=pi_hat)] * len(xsi.columns), axis=1, ignore_index=True)
    price_df.columns = xsi.columns
    if mu == 1:
        price_imports_finaldemand = xsi.mul(np.log(price_df))
        price_imports_finaldemand = np.exp(price_imports_finaldemand.groupby(level='Sector').sum())
    else:
        price_imports_finaldemand = xsi.mul(price_df**(1-mu))
        price_imports_finaldemand = (price_imports_finaldemand.groupby(level='Sector').sum())**(1/(1-mu))
    price_imports_finaldemand = price_imports_finaldemand.reindex(psi.index, axis=0)
    assert price_imports_finaldemand.shape == psi.shape

    # Price for intermediate sectors goods, final demand (price index)
    if sigma == 1:
        price_index = (betai_hat.xs('FRA', level='Country')).mul(psi.mul(np.log(price_imports_finaldemand)))
        price_index = np.exp(price_index.sum(axis=0))
    else:
        price_index = (betai_hat.xs('FRA', level='Country')).mul(psi.mul(price_imports_finaldemand**(1-sigma)))
        price_index = price_index.sum(axis=0)**(1/(1-sigma))
    assert price_index.shape == (C,)

    # Final demand
    price_df = pd.concat([pd.DataFrame(index=xsi.index, data=pi_hat)] * price_imports_finaldemand.shape[1], axis=1, ignore_index=True)
    price_df.columns = price_imports_finaldemand.columns
    # price_df = price_df.unstack(0)
    # price_df.columns = price_df.columns.get_level_values(1)

    # Here, we concatenate directly rows, so we can do what we are writing here
    tmp_priceindex = pd.concat([price_index.to_frame().T] * len(price_df.index), axis=0, ignore_index=True)
    tmp_priceindex.index = price_df.index
    tmp_PSigmaY = pd.concat([PSigmaY] * len(price_df.index), axis=0, ignore_index=True)
    tmp_PSigmaY.index = price_df.index

    # price of imports is the same for each country, by definition of the nest, so we replicate it across rows
    unique_countries = price_df.index.get_level_values(0).unique()
    replicated_priceimports = []
    for country in unique_countries:
        df_copy = price_imports_finaldemand.copy()
        df_copy.index = pd.MultiIndex.from_product([[country], df_copy.index], names=['Country', 'Sector'])
        replicated_priceimports.append(df_copy)
    tmp_priceimports = pd.concat(replicated_priceimports, axis=0)
    # reindex tmp_priceimports so that it follows the same order as tmp_priceindex
    tmp_priceimports = tmp_priceimports.reindex(tmp_priceindex.index)

    final_demand = betai_hat.mul(tmp_PSigmaY.mul((tmp_priceindex **(sigma)).mul((tmp_priceimports ** (mu - sigma)).mul(price_df ** (-mu)))))

    # Intermediate demand
    assert price_imports.index.equals(Domestic.index), "price_imports index and Domestic index are not the same while they should be"
    # the price imports is the same for all sectors used, whether they come from the domestic or other countries. This is why we concatenate across columns.
    replicated_priceimports = []
    for country in unique_countries:
        df_copy = price_imports.copy()
        df_copy.columns = pd.MultiIndex.from_product([[country], df_copy.columns], names=['Country', 'Sector'])
        replicated_priceimports.append(df_copy)
    tmp_priceimports = pd.concat(replicated_priceimports, axis=1)
    # reindex tmp_priceimports so that it follows the same order as tmp_priceindex
    tmp_priceimports = tmp_priceimports.reindex(Domestic.columns, axis=1)


    tmp_price_intermediate = pd.concat([price_intermediate] * tmp_priceimports.shape[1], axis=1, ignore_index=True)  # this price is shared across all sectors
    tmp_price_intermediate.columns = tmp_priceimports.columns
    # In the next lines, we see again the importance of having pi_hat ordered in the same way as Domestic, and Domestic having index and columns ordered similarly
    price_df_origin = pd.concat([pd.DataFrame(index=tmp_priceimports.index, data=pi_hat)] * tmp_priceimports.shape[1], axis=1, ignore_index=True)
    price_df_origin.columns = tmp_priceimports.columns
    price_df_used = pd.concat([pd.DataFrame(columns=tmp_priceimports.columns, data=pi_hat.reshape(1, -1))] * tmp_priceimports.shape[0], axis=0, ignore_index=True)
    price_df_used.index = tmp_priceimports.index
    tmp_yi = pd.concat([pd.DataFrame(index=tmp_priceimports.index, data=yi_hat)] * tmp_priceimports.shape[1], axis=1, ignore_index=True)
    tmp_yi.columns = tmp_priceimports.columns

    intermediate_demand = (price_df_origin**theta).mul((tmp_price_intermediate**(epsilon-theta)).mul((tmp_priceimports**(delta-epsilon)).mul((price_df_used**(-delta)))))
    intermediate_demand = intermediate_demand.mul(tmp_yi)

    ### Residuals
    # check that the order of the different arrays does match !!
    if singlefactor:
        pass
    else:
        # Prices
        if theta == 1:
            # res1 = np.log(pd.Series(pi_hat, index=sectors['eta'].index))  - (sectors['eta'] * np.log(pd.Series(vi_hat, index=sectors['eta'].index)) + (1-sectors['eta']) * np.log(price_intermediate))
            res1 = np.log(pi_hat) - (sectors['eta'].to_numpy() * np.log(vi_hat.to_numpy()) + (1 - sectors['eta']).to_numpy() * np.log(price_intermediate.squeeze().to_numpy()))
        else:
            res1 = pi_hat**(1-theta) - (sectors['eta'].to_numpy() * vi_hat**(1-theta) + (1 - sectors['eta']).to_numpy() * (price_intermediate.squeeze().to_numpy())**(1-theta))

        # Quantities
        phi = sectors.loc[:,sectors.columns.str.contains('phi_')]
        assert phi.shape == final_demand.shape
        res2 = yi_hat - (final_demand.mul(phi).sum(axis=1) + Delta.mul(intermediate_demand).sum(axis=0)).to_numpy()  # TODO: check if the sum is done correctly (depends on equilibrium condition)

        # World GDP
        price_index = price_index.to_frame().T
        price_index.columns = PSigmaY.columns
        res3 = (share_GNE.mul(PSigmaY.mul(price_index)).sum(axis=1)).to_numpy() - 1

        # Definition of GDP in other country
        # TODO: this line should be modified if I include more countries than just two
        li_hat = li_hat.to_frame().xs(domestic_country, level='Country', axis=0)
        ki_hat = ki_hat.to_frame().xs(domestic_country, level='Country', axis=0)
        revenue = sectors.loc[:, sectors.columns.str.contains('rev_')]
        rev_labor = revenue['rev_labor'].xs(domestic_country, level='Country', axis=0).to_frame().rename(columns={'rev_labor': 0})
        rev_capital = revenue['rev_capital'].xs(domestic_country, level='Country', axis=0).to_frame().rename(columns={'rev_capital': 0})
        res4 = rev_labor.mul((li_hat.mul(wi_hat.to_frame().xs(domestic_country, level='Country', axis=0)))).sum(axis=0) + rev_capital.mul((ki_hat.mul(ri_hat.to_frame().xs(domestic_country, level='Country', axis=0)))).sum(axis=0) - PSigmaY.mul(price_index)[domestic_country]
        res4 = res4.to_numpy()

    res = np.concatenate([res1, res2, res3, res4])
    return res

def residuals_wrapper(lvec):
    # Call the original function but only return the first output
    res = residuals(lvec, li_hat,ki_hat,betai_hat,theta,sigma,epsilon,delta,mu,sectors, xsi, psi, Omega, Domestic, Delta, share_GNE, singlefactor=False, domestic_country='FRA')
    print(np.linalg.norm(res))
    # print(lvec)
    return res


filename = "outputs/calib_france.xlsx"
fileshocks = "data_deep/shocks_interventions_large_demand.xlsx"

theta,sigma,epsilon,delta,mu = 1,1,1,1,1

calib = CalibOutput.from_excel(filename)
sectors, share_emissions, xsi, psi, Omega, Gamma, Domestic, Delta, share_GNE, descriptions = calib.sectors, calib.share_emissions, calib.xsi, calib.psi, calib.Omega, calib.Gamma, calib.Domestic, calib.Delta, calib.share_GNE, calib.descriptions

demand_shocks = pd.read_excel(fileshocks, index_col=0, header=0)

# for col in demand_shocks.columns:
#     logging.info(f"Shock {col}")
col = demand_shocks.columns[0]
ki_hat = pd.Series(index=sectors.index, data=1)
li_hat = pd.Series(index=sectors.index, data=1)

singlefactor=False
domestic_country = 'FRA'
N = len(sectors)

# new_index = pd.MultiIndex.from_product([['FRA'], demand_shocks.index], names=['Country', 'Sector'])
# demand_shocks_multi = demand_shocks.set_index(new_index)
# demand_shocks_multi = demand_shocks_multi.reindex(sectors.index, fill_value=1)
# betai_hat = demand_shocks_multi[col]

unique_countries = sectors.index.get_level_values('Country').unique()
replicated_priceimports = []
for country in unique_countries:
    df_copy = demand_shocks.copy()
    df_copy.index = pd.MultiIndex.from_product([[country], df_copy.index], names=['Country', 'Sector'])
    replicated_priceimports.append(df_copy)
demand_shocks = pd.concat(replicated_priceimports, axis=0)
demand_shocks = demand_shocks.reindex(sectors.index)
betai_hat = demand_shocks[col]


initial_guess = np.zeros(2*N + 2)
# initial_guess[0] = -1e-1
# res = residuals(initial_guess, li_hat,ki_hat,betai_hat,theta,sigma,epsilon,delta,mu,sectors, xsi, psi, Omega, Domestic, Delta, share_GNE, singlefactor=False, domestic_country='FRA')


def create_callback_every_10_iterations(func):
    # Closure to hold state - iteration count and previous x values for comparison
    iter_count = {'count': 0}

    def callback(xk):
        # Access and update the iteration count
        iter_count['count'] += 1

        # Every 10 iterations, print some metrics
        if iter_count['count'] % 10 == 0:
            # Assuming 'func' is your function to solve, showing change in function value might be a useful metric
            print(f"Iteration {iter_count['count']}: x = {xk}, f(x) = {func(xk)}")

    return callback

# Solve for zeros using fsolve
# lvec_eq_shocked_single = root(residuals_wrapper, initial_guess, method='lm')
# lvec_eq_shocked_single = fsolve(residuals_wrapper, initial_guess, full_output=True)
