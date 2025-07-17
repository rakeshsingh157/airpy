# utils.py
import gdown

def download_model():
    url = "https://drive.google.com/uc?id=1C5PtEhg3hLq-xDznd1-ylh1k9kX11Dqt"
    output = "rural_aqi_model.pkl"
    gdown.download(url, output, quiet=False)
