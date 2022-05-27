# %%
from webhelper import get_data_from_web
import matplotlib.pyplot as plt

EUMELANIN_EXTCOEFF_URL = "https://omlc.org/spectra/melanin/eumelanin.html"
PHEOMELANIN_EXTCOEFF_URL = "https://omlc.org/spectra/melanin/pheomelanin.html"

eu_data = get_data_from_web(EUMELANIN_EXTCOEFF_URL).T
pheo_data = get_data_from_web(PHEOMELANIN_EXTCOEFF_URL).T
# %%
print(eu_data)
print(type(eu_data))
# %% PLOT LINEAR-LINEAR GRAPHS
# %%
