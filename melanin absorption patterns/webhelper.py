from bs4 import BeautifulSoup
import requests
import numpy as np

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}

def parse_data_tsv(data_string):
    data = []
    data_rows = data_string.split('\n')[6:-1]
    for row in data_rows:
        unit = row.split('\t')
        data.append(unit)
    # convert to float, transpose into 3 rows
    return np.array(data).astype(np.float)

def get_data_from_web(data_url):
    data_src = requests.get(data_url, headers=HEADERS, verify=False)
    data_soup = BeautifulSoup(data_src.text, "html.parser")
    data_txt = data_soup.find("pre").text
    return parse_data_tsv(data_txt)