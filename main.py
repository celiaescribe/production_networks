# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os
from matplotlib import pyplot as plt
import json
import pickle


def preference_change_solver(desired_changes, df, invert=True, fs=False, initial_guess=None):
    sectors = df.sector_shortdescription.to_list()
    initial_budget_shares = df['psi'].tolist()

    if fs:
        # TODO: not working super well because of convergence issues I think
        def to_solve(pref_vector):
            term_sum = np.sum(initial_budget_shares * pref_vector)
            calculated_changes = pref_vector - term_sum
            desired_vector = np.array([desired_changes.get(sector, 0) for sector in sectors])
            return calculated_changes - desired_vector

        # Initial guess for preference changes
        if initial_guess is None:
            initial_guess = np.array([desired_changes.get(sector, 0) for sector in sectors])

        # Use fsolve to find the preference changes
        solved_prefs = fsolve(to_solve, initial_guess)
        return solved_prefs

    else:
        # Create matrix B
        B = np.tile(initial_budget_shares, (len(sectors), 1))

        # Identity matrix I
        I = np.eye(len(sectors))

        if invert:
            # Inverse of (I - B)
            try:
                inv_matrix = np.linalg.inv(I - B)
            except np.linalg.LinAlgError:
                raise ValueError("The matrix (I - B) is not invertible.")

            # Creating the desired changes vector
            dlog_b = np.array([desired_changes.get(sector, 0) for sector in sectors])

            # Solve for d log x
            dlog_x = np.dot(inv_matrix, dlog_b)
            return dlog_x
        else:
            dlog_x = np.array([desired_changes.get(sector, 0) for sector in sectors])
            dlog_b = np.dot(I-B,dlog_x)

            return dlog_b

def calibrate_BEA504_hat_fun(theta, sigma, epsilon):
    print("%% I %%%%%%%%%%%%%%%%%%%%%% Import DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # final consumption
    pfi1 = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012cleaned', usecols='PK', nrows=392).to_numpy().flatten()
    pfi1[np.isnan(pfi1)] = 0
    pfi2 = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012cleaned', usecols='OX', nrows=392).to_numpy().flatten()
    pfi2[np.isnan(pfi2)] = 0
    pfi = pfi1 - pfi2
    pfi[pfi < 0] = 0
    pfi = pfi[np.r_[0:280, 281:len(pfi)]]  # get rid of "Customs duties"

    # IO
    OmegaRaw = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012cleaned', usecols='B:OC', nrows=392).to_numpy()
    OmegaRaw[np.isnan(OmegaRaw)] = 0
    OmegaRaw[OmegaRaw < 0] = 0
    OmegaRaw = OmegaRaw.T  # Transpose OmegaRaw
    OmegaRaw = np.delete(OmegaRaw,(281), axis=1)  # delete column 281
    OmegaRaw = np.delete(OmegaRaw,(281), axis=0)  # delete row 281
    # OmegaRaw = OmegaRaw[[:281,282:],[:281,282:]]
    # OmegaRaw = OmegaRaw[np.r_[0:280, 281:len(OmegaRaw)], np.r_[0:280, 281:len(OmegaRaw)]]  # get rid of "Customs duties"

    pxij = OmegaRaw
    pmXi = np.sum(pxij, axis=1)  # sum over columns

    # total output
    pyi = np.sum(pxij, axis=0) + pfi  # sum over rows

    # number of sectors
    N = len(pxij)

    # normalized the matrix
    Gamma = np.array([pxij[i, :] / pyi[i] for i in range(N)])

    # Labor income share in Value Added
    empcomp = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012cleaned', usecols='B:OC').to_numpy()
    empcomp =empcomp[406,:].flatten()
    va = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012cleaned', usecols='B:OC').to_numpy()
    va = va[409,:].flatten()
    wli_over_vhi = (empcomp / va).T
    wli_over_vhi = np.delete(wli_over_vhi,(281), axis=0)  # delete row 281

    # Load sector description
    sector_info = pd.read_excel('data_deep/IOUse_Before_Redefinitions_PRO_DET.xlsx', sheet_name='2012', usecols='A:B')
    sector_info = sector_info[5:397]
    sector_info = pd.DataFrame(sector_info)
    sector_info.columns = ['A', 'B']
    sector_code = sector_info['A'].apply(lambda x: str(x) if pd.notna(x) else '').tolist()
    sector_shortdescription = sector_info['B'].tolist()
    sector_code = sector_code[0:281] + sector_code[282:]
    sector_shortdescription = sector_shortdescription[0:281] + sector_shortdescription[282:]

    # II - Calibrate Parameters
    print("%% II %%%%%%%%%%%%%%%%%%%%%% Calibrate Parameters  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Omega_ij = share of j in total intermediate inputs cost by i
    Omega = np.diag(1 / pmXi) @ pxij

    # eta_i = value added share (of revenue) in sector i
    eta = 1 - pmXi / pyi

    # phi_i = share of final demand i in total output of i
    phi = pfi / pyi

    # Delta_ij = expenditure on j by i as a share of total production of j
    Delta = pxij @ np.diag(1 / pyi)

    # psi_i = share of final demand i in to total final demand
    psi = pfi / np.sum(pfi)

    # gamma_i = labor income share in value added in sector i
    gamma = wli_over_vhi

    # sector labor income over GDP
    va = pyi - pmXi
    # wli_over_GDP
    lambda_ = wli_over_vhi * va / np.sum(pfi)
    # rli_over_GDP
    rho = (1 - wli_over_vhi) * va / np.sum(pfi)

    # III - Test: Equilibrium
    print("%% III %%%%%%%%%%%%%%%%%%%%%% Test: Equilibrium %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # ... (remaining code to be translated, including optimization and saving results)

    # IV - Networks Stats
    print("%% IV %%%%%%%%%%%%%%%%%%%%%% Networks Stats %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Outdegree
    outdegree = Gamma.sum(axis=1)

    # Centrality
    centrality = np.linalg.inv(np.eye(N) - Gamma) @ (pfi / sum(pfi))

    # Upstreamness
    upstreamness = np.linalg.inv(np.eye(N) - np.diag(1 / pyi) @ Gamma.T @ np.diag(pyi)) @ np.ones(N)

    # Domar weights
    domar = pyi / sum(pfi)

    # V - Save
    print("%% V %%%%%%%%%%%%%%%%%%%%%% Save  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Preparing data to save
    Data = {
        'N': N,
        'pfi': pfi,
        'pxij': pxij,
        'pmXi': pmXi,
        'pyi': pyi,
        'wli_over_vhi': wli_over_vhi,
        'Gamma': Gamma,
        'sector_code': sector_code,
        'sector_shortdescription': sector_shortdescription
    }

    Calibration = {
        'N': N,
        'Omega': Omega,
        'eta': eta,
        'phi': phi,
        'Delta': Delta,
        'psi': psi,
        'gamma': gamma,
        'lambda': lambda_,
        'rho': rho
    }

    with open('data_python/US_BEA504_hat_data.json', "wb") as pickle_file:
        pickle.dump(Data, pickle_file)

    with open('data_python/US_BEA504_hat_calibration.json', "wb") as pickle_file:
        pickle.dump(Calibration, pickle_file)

    # # Save as JSON for Python-friendly format
    # with open('data_python/US_BEA504_hat.json', 'w') as f:
    #     json.dump({'Cali': Cali, 'Data': Data}, f, indent=4)

    # Calculating Kappa
    Kappa = np.linalg.inv(np.eye(N) - Delta)

    # Prepare DataFrame for table_data
    table_data = pd.DataFrame({
        'sector_code': sector_code,
        'sector_shortdescription': sector_shortdescription,
        'eta': eta,
        'phi': phi,
        'psi': psi,
        'gamma': gamma,
        'lambda': lambda_,
        'rho': rho
    })

    # Save to Excel
    filename_export = 'export/US_BEA504_data.xlsx'
    table_data.to_excel(filename_export, sheet_name='Para', index=False)

    with pd.ExcelWriter(filename_export, mode='a', engine='openpyxl') as writer:
        pd.DataFrame(Delta, index=sector_code, columns=sector_code).to_excel(writer, sheet_name='Delta', index=False)
        pd.DataFrame(Omega, index=sector_code, columns=sector_code).to_excel(writer, sheet_name='Omega')
        pd.DataFrame(Kappa, index=sector_code, columns=sector_code).to_excel(writer, sheet_name='Kappa')

    pd.DataFrame(Delta, index=sector_code, columns=sector_code).to_excel(filename_export, sheet_name='Delta')
    pd.DataFrame(Omega, index=sector_code, columns=sector_code).to_excel(filename_export, sheet_name='Omega')
    pd.DataFrame(Kappa, index=sector_code, columns=sector_code).to_excel(filename_export, sheet_name='Kappa')

    print("Data saved successfully.")

def RES_hat_eq(lvec, zi_hat: np.ndarray, li_hat, ki_hat, betai_hat, theta, sigma, epsilon, N, Omega, eta, gamma, phi, Delta,
                  psi):
    ### All vectors like li_hat should be of size (N,)
    # Load the unknown
    vec = np.exp(lvec)

    # Load price
    pi_hat = vec[0:N]

    # Load quantity
    yi_hat = vec[N:2 * N]

    # GDP
    PSigmaY_hat = vec[2 * N]

    # Some useful variables
    pi_hat_1MoinsEpsi = pi_hat ** (1 - epsilon)
    pi_hat_1MoinsThetai = pi_hat ** (1 - theta)

    # Intermediate Bundle Price
    if epsilon == 1:
        Pmi_hat = np.exp(Omega @ np.log(pi_hat))
    else:
        Pmi_hat = (Omega @ pi_hat_1MoinsEpsi) ** (1 / (1 - epsilon))

    # Value added bundle quantity
    hi_hat = li_hat ** gamma * ki_hat ** (1 - gamma)

    # Value added bundle price
    vi_hat = zi_hat ** ((theta - 1) / theta) * (yi_hat / hi_hat) ** (1 / theta) * pi_hat

    # Wage
    wi_hat = vi_hat * hi_hat / li_hat

    # Rental rate
    ri_hat = vi_hat * hi_hat / ki_hat

    # Final demand
    fi_hat = betai_hat * pi_hat ** (-sigma) * PSigmaY_hat

    # Intermediate demand
    xij_hat = (zi_hat ** (theta - 1) * Pmi_hat ** (epsilon - theta) * pi_hat ** theta * yi_hat) @ (
                pi_hat ** (-epsilon)).T

    # The residuals equation
    # Sector price = marginal cost
    if theta == 1:
        RES1 = np.log(pi_hat) - (-np.log(zi_hat) + eta * np.log(vi_hat) + (1 - eta) * np.log(Pmi_hat))
    else:
        RES1 = pi_hat_1MoinsThetai - zi_hat ** (theta - 1) * (
                    eta * vi_hat ** (1 - theta) + (1 - eta) * Pmi_hat ** (1 - theta))

    # Sector quantity = market clearing
    RES2 = yi_hat - ((phi * fi_hat) + (Delta @ xij_hat).sum(axis=1))

    # Price index
    if sigma == 1:
        RES3 = 0 - np.sum(betai_hat * psi * np.log(pi_hat))
    else:
        RES3 = np.sum(betai_hat * psi * pi_hat ** (1 - sigma)) - 1

    # The residual
    RES = np.concatenate([RES1, RES2, np.array([RES3])])

    # All the variables in a structure
    Equi_hat = {
        'zi': zi_hat,
        'li': li_hat,
        'ki': ki_hat,
        'GDP': PSigmaY_hat,
        'domar': pi_hat * yi_hat / PSigmaY_hat,
        'pyi': pi_hat * yi_hat,
        'pfi': pi_hat * fi_hat,
        'wli': wi_hat * li_hat,
        'rki': ri_hat * ki_hat,
        'wi': wi_hat,
        'ri': ri_hat,
        'vi': vi_hat,
        'hi': hi_hat,
        'vhi': vi_hat * hi_hat,
        'pi': pi_hat,
        'yi': yi_hat
    }

    return RES, PSigmaY_hat, Equi_hat


def exercice_hat_fun_nicefigure(theta, sigma, epsilon, cap, califile, excelfile, fig=None):
    # I - Load Data
    print("%% I %%%%%%%%%%%%%%%%%%%%%% Load Data   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    filename = f'data_matlab/{califile}'
    # Load your data here
    # Example: Data = loadmat(filename) # Assuming loadmat is a function to load MATLAB data
    # For this example, I will create a placeholder for Data and Cali
    Data = {'sector_code': None, 'sector_shortdescription': None}  # Placeholder
    Cali = {'N': None, 'Omega': None, 'eta': None, 'phi': None, 'Delta': None, 'psi': None,
            'gamma': None}  # Placeholder

    print(f'load: {filename}')

    # Data
    sector_code = Data['sector_code']
    sector_shortdescription = Data['sector_shortdescription']

    # Calibrated parameters
    N = Cali['N']
    Omega = Cali['Omega']
    eta = Cali['eta']
    phi = Cali['phi']
    Delta = Cali['Delta']
    psi = Cali['psi']
    gamma = Cali['gamma']

    # III - Compute Effect of Shock
    print("%% III %%%%%%%%%%%%%%%%%%%%%% Compute Effect of Shock  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print(f'Using shocks from {excelfile}.xlsx')

    options = {'disp': True}

    Alphabet = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Load the list of supply shocks
    filename = f'data_deep/{excelfile}_laborsupply_celia.xlsx'
    df = pd.read_excel(filename)
    header = df.columns.tolist()
    confinement = header[1:]

    # Load the list of demand shocks
    filename = f'data_deep/{excelfile}_demand_celia.xlsx'
    df = pd.read_excel(filename)
    header = df.columns.tolist()
    demand_header = header[1:]

    if fig is not None and fig == 1:
        figsize_inches = (0.6 * 100, 0.6 * 100)  # Width and height in inches
        fig = plt.figure(figsize=figsize_inches)

        # You can use a Python list for the legend, similar to MATLAB's cell array
        # Replace 'length(confinement)+1' with the actual number based on your data
        legend_size = len(confinement)
        Legend = [None] * legend_size

    for c in range(len(confinement)):
        for d in range(len(demand_header)):
            print(f'------ supply: {confinement[c]} ----------')
            print(f'------ demand: {demand_header[d]} ----------')

            # # Load the shock
            # if c > 0:
            #     # Load supply shock data for confinement[c]
            #     share_shock = supply_shock_data(excelfile, Alphabet[c], N)  # Placeholder
            #     shocks = np.minimum(share_shock, cap)
            #
            # if d > 0:
            #     # Load demand shock data for demand_header[d]
            #     demand_shock = demand_shock_data(excelfile, Alphabet[d], N)  # Placeholder


if __name__ == '__main__':
    # calibrate_BEA504_hat_fun(theta=1, sigma=1, epsilon=1)
    # exercice_hat_fun_nicefigure(theta=1, sigma=1, epsilon=1, cap=1, califile=None, excelfile='Data_US_BEA405_DEC2020')

    df = pd.read_excel('export/US_BEA504_data 2.xlsx')
    df = pd.DataFrame(df)
    desired_changes = {'Motor vehicle and parts dealers': - 0.2, 'Petroleum refineries': -0.2}
    desired_changes = {'Motor vehicle and parts dealers': - 0.2, 'Rail transportation': 0.05}
    sectors = df.sector_shortdescription.to_list()
    desired_changes2 = {sector: desired_changes.get(sector, 0.0066469) for sector in sectors}

    solved_prefs0 = preference_change_solver(desired_changes, df, invert=True, fs=True, initial_guess= np.array([desired_changes.get(sector, 0.005) for sector in sectors]))
    solved_prefs = preference_change_solver(desired_changes2, df, invert=False)

    # df[df['sector_shortdescription'].str.contains('vehicle')]
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


    # Reshape dataframe df, from 2 index A and B and 2 columns C and D into multiindex of size A,B,C,D
    



