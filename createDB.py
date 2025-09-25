import yfinance as yf
import pandas as pd
from variable_function import add_synthetic_variable  # ta fonction PCA + cluster
from variable_function import add_rsi, add_macd, add_emas, add_bollinger_bands, add_volatility # Import fonction

# dataset Gold via api yfinance
gold_data = yf.download("GC=F", start="2000-01-01", end="2025-09-01", interval="1d")

# Colonne prix renommée en 'Price' pour cohérence avec les fonctions
gold_data.rename(columns={'Close':'Price'}, inplace=True)
print(gold_data.head())

# Appeler les fonction
gold_data = add_rsi(gold_data)
gold_data = add_macd(gold_data)
gold_data = add_emas(gold_data)
gold_data = add_bollinger_bands(gold_data)
gold_data = add_volatility(gold_data)
gold_data = add_synthetic_variable(gold_data)

# Focus PCA
print(gold_data["Synth_PCA"])
import matplotlib.pyplot as plt

print("Nombre de valeurs positives :", (gold_data["Synth_PCA"] > 0).sum())
print("Nombre de valeurs négatives :", (gold_data["Synth_PCA"] < 0).sum())
print("Nombre de valeurs nulles :", (gold_data["Synth_PCA"] == 0).sum()) # on a bien du + et -



# Afficher le nombre de lignes non-NaN par colonne
valid_counts = gold_data.notna().sum()
print(valid_counts) # Synth_PCA a le plus de NaN (car PCA sur bcp de variable), same pour Synth_Cluster

# DATACLEAN + head + tail
gold_data.dropna(inplace=True)  # Supprimer les lignes avec des valeurs manquantes
valid_counts = gold_data.notna().sum() # À cause de la suppression des NaN, on perd presque 2k lignes du à PCA et CLUSTER.
print(valid_counts)

print(gold_data.head())
print(gold_data.tail())








