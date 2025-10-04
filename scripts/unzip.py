from zipfile import ZipFile
with ZipFile("../inputs/predicting-the-price-of-diamond.zip") as zip_ref:
    zip_ref.extractall("../inputs")

import os 
os.remove("../inputs/predicting-the-price-of-diamond.zip")