# %%
from webhelper import get_data_from_web
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

EUMELANIN_EXTCOEFF_URL = "https://omlc.org/spectra/melanin/eumelanin.html"
PHEOMELANIN_EXTCOEFF_URL = "https://omlc.org/spectra/melanin/pheomelanin.html"

eu_data = get_data_from_web(EUMELANIN_EXTCOEFF_URL).T
pheo_data = get_data_from_web(PHEOMELANIN_EXTCOEFF_URL).T
# %%
print(eu_data)
print(type(eu_data))
# %% PLOT LINEAR-LINEAR GRAPHS OF mg/ml
plt.plot(eu_data[0], eu_data[1], label="eumelanin")
plt.plot(pheo_data[0], pheo_data[1], label="pheomelanin")
plt.legend()
# %% PLOT LINEAR-LINEAR GRAPHS OF mol/L
plt.plot(eu_data[0], eu_data[2], label="eumelanin")
plt.plot(pheo_data[0], pheo_data[2], label="pheomelanin")
plt.legend()
# %% PLOT SEMILOG GRAPHS OF mg/ml
plt.plot(eu_data[0], np.log10(eu_data[1]), label="Eumelanin")
plt.plot(pheo_data[0], np.log10(pheo_data[1]), label="Pheomelanin")
plt.xlim(350, 700)
plt.legend()
# %% PLOT SEMILOG GRAPHS OF mol/L
plt.plot(eu_data[0], np.log10(eu_data[2]), label="Eumelanin")
plt.plot(pheo_data[0], np.log10(pheo_data[2]), label="Pheomelanin")
plt.xlim(350, 700)
plt.legend()
# %% Linear Regression Function
def lin_model(data):
    wavelen = data[0]
    mg_ml = data[1]
    mol_l = data[2]

    m, b, r, p, s = linregress(wavelen, mg_ml)
    linmodel_mgml = (m, b, r, p, s)

    m, b, r, p, s = linregress(wavelen, mol_l)
    linmodel_moll = (m, b, r, p, s)

    return linmodel_mgml, linmodel_moll

def lin_data(x, model):
    return model[0] * x + model[1]


# %% Get linear models for sets of data
eu_1, eu_2 = lin_model(eu_data)

# %% Plot data for mgml
plt.plot(eu_data[0], eu_data[1], 'bo', eu_data[0], lin_data(eu_data[0], eu_1), '--k')
plt.xlim(350, 700)
# %% Plot data for moll
plt.plot(eu_data[0], eu_data[2], 'bo', eu_data[0], lin_data(eu_data[0], eu_2), '--k')
plt.xlim(350, 700)
# %% Create semilog linear model
def log_lin_model(data):
    wavelen = data[0]
    mg_ml = np.log10(data[1])
    mol_l = np.log10(data[2])

    m, b, r, p, s = linregress(wavelen, mg_ml)
    linmodel_mgml = (m, b, r, p, s)

    m, b, r, p, s = linregress(wavelen, mol_l)
    linmodel_moll = (m, b, r, p, s)

    return linmodel_mgml, linmodel_moll
# %% get semilog models
semilog_eu_1, semilog_eu_2 = log_lin_model(eu_data)
# %% plot semilog graphs with linear regression
plt.plot(eu_data[0], np.log10(eu_data[1]), 'bo', eu_data[0], lin_data(eu_data[0], semilog_eu_1), '--k')
plt.xlim(350, 700)
# %%
print (semilog_eu_1[0], semilog_eu_2[0])
# %%
plt.plot(eu_data[0], np.log10(eu_data[2]), 'bo', eu_data[0], lin_data(eu_data[0], semilog_eu_2), '--k')
plt.xlim(350, 700)
# %%
