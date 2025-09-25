
"""
This script downloads historical gold price data using yfinance, computes daily and 2-week shocks,
calculates rolling volatility and growth thresholds, generates binary targets for large shocks,
and visualizes the price with Bollinger Bands.
Usage: Run the script directly to fetch data, compute features, and display the plot.
"""


import yfinance as yf
import matplotlib.pyplot as plt

gold = yf.download("GC=F", start="1980-01-01", end="2025-09-01")

# Ajout de la variation quotidienne (valeur absolue)
gold['Daily Change'] = gold['Close'].diff()

# Ajout de la variation quotidienne en pourcentage
gold['Daily Change %'] = gold['Close'].pct_change() * 100

print(gold[['Close', 'Daily Change', 'Daily Change %']])

# Choc à 2 semaines (10 jours ouvrés)
gold['Shock_2w'] = gold['Close'].shift(-10) - gold['Close']
gold['Shock_2w %'] = (gold['Close'].shift(-10) - gold['Close']) / gold['Close'] * 100

# Afficher les 25 dernières lignes
print(gold[['Close', 'Daily Change', 'Daily Change %', 'Shock_2w', 'Shock_2w %']].tail(25))

# Calcul de la volatilité mobile (écart-type sur 30 jours des variations quotidiennes en %)
gold['Volatility_30d'] = gold['Daily Change %'].rolling(window=30).std()

# Seuil variable : 2 fois la volatilité mobile
gold['Seuil_variable'] = 2 * gold['Volatility_30d']

# Cible binaire : 1 si le choc à 2 semaines (en %) dépasse le seuil variable, 0 sinon
gold['Target_Shock'] = (gold['Shock_2w %'].abs() >= gold['Seuil_variable']).astype(int)

# Afficher un extrait pour vérifier
print(gold[['Close', 'Shock_2w %', 'Seuil_variable', 'Target_Shock']].tail(25))


# Calcul du taux de croissance moyen absolu sur 30 jours
gold['MeanGrowth_30d'] = gold['Daily Change %'].abs().rolling(window=30).mean()

# Seuil variable : par exemple, 2 fois ce taux de croissance moyen
gold['Seuil_growth'] = 2 * gold['MeanGrowth_30d']

# Cible binaire : 1 si le choc à 2 semaines dépasse ce seuil, 0 sinon
gold['Target_Shock_growth'] = (gold['Shock_2w %'].abs() >= gold['Seuil_growth']).astype(int)

# Affichage pour vérification
print(gold[['Close', 'Shock_2w %', 'Seuil_growth', 'Target_Shock_growth']].tail(25))

# ...existing code...

# Calcul des bandes de Bollinger (20 jours, 2 écarts-types)
gold['Bollinger_MA20'] = gold['Close'].rolling(window=20).mean()
gold['Bollinger_STD20'] = gold['Close'].rolling(window=20).std()
gold['Bollinger_Upper'] = gold['Bollinger_MA20'] + 2 * gold['Bollinger_STD20']
gold['Bollinger_Lower'] = gold['Bollinger_MA20'] - 2 * gold['Bollinger_STD20']

# Tracer le graphique
plt.figure(figsize=(14, 7))
plt.plot(gold['Close'], label='Prix de clôture')
plt.plot(gold['Bollinger_MA20'], label='Moyenne mobile 20j', color='orange')
plt.plot(gold['Bollinger_Upper'], label='Bollinger Upper', linestyle='--', color='green')
plt.plot(gold['Bollinger_Lower'], label='Bollinger Lower', linestyle='--', color='red')
plt.title("Prix de l'or et bandes de Bollinger")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.tight_layout()
plt.show()