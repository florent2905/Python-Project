import kagglehub # kaggle pour telecharger les datasets
import os
import shutil
import pandas as pd

# dataset kaggle sentiment analyses
dataset_name = "ankurzing/sentiment-analysis-in-commodity-market-gold"
download_path = kagglehub.dataset_download(dataset_name)
print("Dataset téléchargé dans :", download_path)

# csv file path
csv_file = None
for root, dirs, files in os.walk(download_path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)
            print("CSV trouvé :", csv_file)
            break
    if csv_file:
        break

if not csv_file:
    raise FileNotFoundError("Aucun fichier CSV trouvé dans le dataset.")

# Copier le fichier CSV dans le dossier actuel pour un accès plus facile
current_folder = os.getcwd()  # récupère le dossier où tu es en train de travailler
destination_path = os.path.join(current_folder, os.path.basename(csv_file))
shutil.copy(csv_file, destination_path)
# Le csv est maintenant dans le dossier actuel

# name df 
sentiment_vader = df = pd.read_csv(destination_path)

# decouverte des donnees
print(sentiment_vader.info())
sentiment_vader.head
sentiment_vader.columns

sentiment_df = sentiment_vader[['Dates', 'News']].copy()
print(sentiment_df.head())


import pandas as pd
import unicodedata
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Lexique VADER
nltk.download('vader_lexicon')

# copier le df pour le modifier et ajouter des colonnes scores de sentiment
sent_df = sentiment_df.copy()
sent_df["sentiment_score"] = 0.0
sent_df["Negative"] = 0.0
sent_df["Neutral"] = 0.0
sent_df["Positive"] = 0.0
print(sent_df.head())

# Initialiser l'analyseur de sentiment VADER
sentiment_analyzer = SentimentIntensityAnalyzer()

# Analyser chaque ligne du DataFrame
for indx, row in sent_df.iterrows():
    try:
        # Normaliser le texte pour enlever les caractères spéciaux
        text = unicodedata.normalize('NFKD', row['News'])
        scores = sentiment_analyzer.polarity_scores(text)

        # Remplir les colonnes
        sent_df.at[indx, 'sentiment_score'] = scores['compound']
        sent_df.at[indx, 'Negative'] = scores['neg']
        sent_df.at[indx, 'Neutral'] = scores['neu']
        sent_df.at[indx, 'Positive'] = scores['pos']
    except TypeError:
        print(f"Problème de texte à l'index {indx}: {row['News']}")
        continue

# Convertir la colonne 'Dates' en datetime
sent_df['Dates'] = pd.to_datetime(sent_df['Dates'], errors='coerce')  # erreurs deviennent NaT

# Supprimer les lignes où la conversion a échoué
sent_df = sent_df.dropna(subset=['Dates'])

# Trier par date et réinitialiser l'index
sent_df = sent_df.sort_values('Dates').reset_index(drop=True)
sent_df = sent_df[['Dates', 'News', 'sentiment_score']]

print(sent_df.info())
print(sent_df.head())

import yfinance as yf
import pandas as pd
data_gold = yf.download("GC=F", start="2000-01-01", end="2025-09-01", interval="1d")
print(data_gold.info())
print(data_gold.head())

#####################################################
############## MERGE DES DEUX DATAFRAMES ############
###################################################

import pandas as pd

# Assurer que les dates sont en datetime
sent_df['Dates'] = pd.to_datetime(sent_df['Dates'], errors='coerce')
data_gold.index = pd.to_datetime(data_gold.index)

# Extraire la colonne Close et la renommer
gold_close = data_gold.xs('GC=F', level=1, axis=1)['Close'].reset_index()  # level=1 correspond au ticker
gold_close = gold_close.rename(columns={'Close': 'Gold_Close'})

# garder seulement les colonnes nécessaires
sent_clean = sent_df[['Dates', 'News', 'sentiment_score']].copy()

# merger les deux dataframes sur les dates
merged_df = pd.merge(sent_clean, gold_close, left_on='Dates', right_on='Date', how='inner')

# Supprimer colonne Date dupliquée et mettre Dates comme index
merged_df.drop(columns=['Date'], inplace=True)
merged_df.set_index('Dates', inplace=True)
merged_df.sort_index(inplace=True)

print(merged_df.head())
print(merged_df.tail())

# graphique 1 : graph séparée classique

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), sharex=True)

# Prix du gold
ax1.plot(merged_df.index, merged_df['Gold_Close'], color='gold')
ax1.set_ylabel("Gold Price (USD)")
ax1.set_title("Gold Price")

# Sentiment
ax2.plot(merged_df.index, merged_df['sentiment_score'], color='blue')
ax2.set_ylabel("Sentiment Score")
ax2.set_title("Sentiment Analysis (VADER)")

plt.xlabel("Date")
plt.tight_layout()
plt.show()

# graphique 2 : fond coloré selon sentiment

import matplotlib.pyplot as plt

# Graphique du prix de l'or et du sentiment
plt.figure(figsize=(14,6))

# Tracer le prix de l'or
plt.plot(merged_df.index, merged_df['Gold_Close'], label='Gold Close Price', color='gold')

# Colorer le fond selon le sentiment
for i in range(len(merged_df)-1):
    if merged_df['sentiment_score'].iloc[i] > 0:
        plt.axvspan(merged_df.index[i], merged_df.index[i+1], color='green', alpha=0.1)
    elif merged_df['sentiment_score'].iloc[i] < 0:
        plt.axvspan(merged_df.index[i], merged_df.index[i+1], color='red', alpha=0.1)

plt.title("Gold Close Price with Sentiment Overlay")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# graphique 3 : double axe y

import matplotlib.pyplot as plt

# Assurer que l'index est datetime
merged_df.index = pd.to_datetime(merged_df.index)


fig, ax1 = plt.subplots(figsize=(14,6))
# axe gauche : Prix de l'or
ax1.plot(merged_df.index, merged_df['Gold_Close'], color='gold', label='Gold Close Price')
ax1.set_xlabel("Date")
ax1.set_ylabel("Gold Price (USD)", color='gold')
ax1.tick_params(axis='y', labelcolor='gold')
ax1.grid(True)

# axe droit : Sentiment Score
ax2 = ax1.twinx()  # créer un second axe y partageant le même x
ax2.plot(merged_df.index, merged_df['sentiment_score'], color='blue', label='Sentiment Score')
ax2.set_ylabel("Sentiment Score", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# titres et légendes
plt.title("Gold Price and Sentiment Score Over Time")
fig.tight_layout()
plt.show()