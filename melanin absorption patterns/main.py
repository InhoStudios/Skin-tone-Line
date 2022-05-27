# %%
from webhelper import get_data_from_web
import numpy as np
import matplotlib.pyplot as plt

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

    # generate fits
    fit1 = np.polyfit(wavelen, mg_ml, 1)
    fit2 = np.polyfit(wavelen, mol_l, 1)

    # create fit functions
    polyld_mgml = np.polyld(fit1)
    polyld_moll = np.polyld(fit2)

    return polyld_mgml, polyld_moll

# %% Plot data with lin fit
eu_1, eu_2 = lin_model(eu_data)
plt.plot(eu_data[0], eu_data[1], 'bo', eu_data[0], eu_1(eu_data[0]), '--k')
plt.xlim(350, 700)

# %%
