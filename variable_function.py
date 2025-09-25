import pandas as pd

# on définit une fonction avec un df de base
# df => Data frame contenant une colonne de prix
# period => nb de période pour le RSI
# price_column => ou se situe le prix


def add_macd(df, price_column='Price', fast_period=12, slow_period=26, signal_period=9):
    
    #Ajoute les colonnes MACD line et Signal line au DataFrame.

    # price_column : str -> colonne sur laquelle calculer le MACD (par défaut 'Price')
    # fast_period : période de l'EMA rapide (par défaut 12)
    # slow_period : période de l'EMA lente (par défaut 26)
    # signal_period : période de l'EMA de la ligne signal (par défaut 9)

    #Retourne : pd.DataFrame avec colonnes 'MACD' et 'Signal_Line'
    
    
    df = df.copy()  # éviter de modifier le dataframe original
    
    # EMA rapides et lentes
    ema_fast = df[price_column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[price_column].ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    df['MACD'] = ema_fast - ema_slow
    
    # Signal line (EMA du MACD)
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    return df



def add_rsi(df, period=14, price_column='Price'):
    
    df = df.copy()  # On copie pour ne pas modifier l'original

    # calcul des variation 
    delta = df[price_column].diff()

    # Gains et pertes
    gain = delta.clip(lower=0)  # remplace les valeurs négatives par 0 # pourquoi remplacer les valeurs negative et positive
    loss = -delta.clip(upper=0) # remplace les valeurs positives par 0

    # Moyenne des gains et pertes sur 'period' jours
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


    # Calcul du RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs)) 

    return df

def add_emas(df, price_column='Price', ema_periods=[20,50,100,200]):
    
    # Ajoute des EMA (Exponential Moving Average) au dataframe.
    # price_column : str -> colonne sur laquelle calculer les EMA (par défaut 'Price')
    # ema_periods : list -> liste des périodes pour lesquelles calculer les EMA
    # Retourne : pd.DataFrame avec les EMA ajoutées
    
    df = df.copy()  # éviter de modifier le dataframe original
    for period in ema_periods:
        col_name = f'EMA{period}'
        df[col_name] = df[price_column].ewm(span=period, adjust=False).mean()  # calcul EMA
    return df



import pandas as pd

def add_bollinger_bands(df, price_column='Price', period=20, std_factor=2):
    
    #Ajoute les Bandes de Bollinger au dataframe.
    
    #df : pd.DataFrame -> DataFrame avec la colonne de prix
    #price_column : str -> colonne de prix à utiliser
    #period : int -> nombre de périodes pour la SMA et l'écart-type
    #std_factor : float -> facteur multiplicatif de l'écart-type
    
    #Retourne : pd.DataFrame avec colonnes SMA, Upper Band, Lower Band
    
    df = df.copy()
    # SMA (Middle Band)
    df['BB_Middle'] = df[price_column].rolling(window=period, min_periods=period).mean()
    # Écart-type
    df['BB_STD'] = df[price_column].rolling(window=period, min_periods=period).std()
    # Upper Band
    df['BB_Upper'] = df['BB_Middle'] + std_factor * df['BB_STD']
    # Lower Band
    df['BB_Lower'] = df['BB_Middle'] - std_factor * df['BB_STD']
    # Supprimer la colonne intermédiaire STD si tu veux
    df.drop(columns=['BB_STD'], inplace=True)
    
    return df

def add_volatility(df, price_column="Price"):
    """
    Ajoute une colonne 'Volatility' qui correspond au rendement centré réduit
    (z-score) et renvoie aussi la volatilité globale de la série.
    """
    df = df.copy()
    # Rendements journaliers
    df["Return"] = df[price_column].pct_change()
    
    # Volatilité globale (écart-type des rendements)
    volatility = df["Return"].std()
    
    # On peut normaliser chaque rendement par la volatilité
    df["Volatility"] = df["Return"] / volatility
    
    return df


def add_synthetic_variable(df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    temp_df = pd.DataFrame(index=df.index)

    # Variables de base
    temp_df["Return"] = df["Price"].pct_change()
    temp_df["Log_Return"] = np.log(df["Price"] / df["Price"].shift(1))
    temp_df["Vol_5d"] = temp_df["Return"].rolling(5).std()
    temp_df["Vol_10d"] = temp_df["Return"].rolling(10).std()
    temp_df["Momentum_5d"] = df["Price"] - df["Price"].shift(5)
    temp_df["Momentum_10d"] = df["Price"] - df["Price"].shift(10)
    temp_df["High_Low"] = df["High"] - df["Low"]

    # Ratio avec protection contre la division par zéro
    temp_df["Return_to_Range"] = np.where(
        temp_df["High_Low"] == 0,
        0,
        temp_df["Return"] / temp_df["High_Low"]
    )

    # Remplacer les NaN initiaux par 0 (sauf les 10 premières lignes qu’on coupe)
    temp_df = temp_df.fillna(0)
    temp_df = temp_df.iloc[10:]  # on sait qu’il faut au moins 10 périodes pour calculer Momentum_10d et Vol_10d

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(temp_df)

    # PCA (1 composante principale)
    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(X_scaled)

    # KMeans (3 clusters : hausse / baisse / neutre)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_values = kmeans.fit_predict(X_scaled)

    # Ajout au DataFrame original
    df.loc[temp_df.index, "Synth_PCA"] = pca_values.ravel()
    df.loc[temp_df.index, "Synth_Cluster"] = cluster_values

    return df
