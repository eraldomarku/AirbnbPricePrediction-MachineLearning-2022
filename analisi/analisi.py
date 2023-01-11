import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sn
import matplotlib.pyplot as plt
import geopy.distance
from sklearn.metrics import accuracy_score
import math
import datetime
from datetime import datetime
from sklearn.impute import KNNImputer



#Carichiamo il dataset senza nan
df = pd.read_csv('df_train_analisi.csv')

#Imposto visualizzate tutte le colonne
pd.options.display.max_columns = df.shape[1]

#Vediamo quali feature sono rimaste
print(df.keys())
print(df.head(20))
feature_rimaste = ['Unnamed: 0', 'log_price', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'city',
       'instant_bookable', 'latitude', 'longitude', 'neighbourhood',
       'number_of_reviews', 'bedrooms', 'beds', 'room_type_importance', 'pool',
       'gym', 'air', 'parking', 'kitchen', 'tv', 'number_of_amenities',
       'bed_type_importance', 'cancellation_policy_importance',
       'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
       'neighbourhood_mean_price']

# Andiamo ad eliminare alcune feature ridondanti che abbiamo ingegnerizzato
# Eliminiamo unnamed0 perchè sono indici
df = df.drop("Unnamed: 0", axis=1)

# "property type" l'abbiamo modificata prima riducendo le categorie da 36 a 5

# "room_type" la possiamo eliminare dato che abbiamo creato "room_type_importance" dove entire_home > private room > shared_room
df = df.drop("room_type", axis=1)

# "accomodates", "bathrooms", la lasciamo come è

# "bed_type" la possiamo eliminare dato che abbiamo creato "bed_type_importance" che assegna ad esempio maggior valore al letto matrimoniale che al divano
df = df.drop("bed_type", axis=1)

# "cancellation_policy" la possiamo eliminare dato che abbiamo creato "cancellation_policy_importance" che assegna valore maggiore alle policy stringenti
df = df.drop("cancellation_policy", axis=1)

# "cleaning_fee" la lasciamo cosi bool

# "city" la lasciamo anche se abbiamo aggiunto un indice sugli affitti delle case per ciascuna città

# "instant_bookable" la lasciamo bool

# "latitude" e "longitude le eliminiamo"
df = df.drop(labels=["longitude", "latitude"], axis=1)

# "neighbourhood" la possiamo eliminare dato che abbiamo creato "neighbourgood_mean_price"
df = df.drop("neighbourhood", axis=1)

# ''number_of_reviews', 'bedrooms', 'beds', 'room_type_importance', 'pool',
#        'gym', 'air', 'parking', 'kitchen', 'tv', 'number_of_amenities',
#        'bed_type_importance', 'cancellation_policy_importance',
#        'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
#        'neighbourhood_mean_price'
# Le lasciamo cosi come sono, numeriche.

#Andiamo a vedere cosa è rimasto
print(df.keys())
print(df.head(20))
feature_rimaste = ['log_price', 'property_type', 'accommodates', 'bathrooms',
       'cleaning_fee', 'city', 'instant_bookable', 'number_of_reviews',
       'bedrooms', 'beds', 'room_type_importance', 'pool', 'gym', 'air',
       'parking', 'kitchen', 'tv', 'number_of_amenities',
       'bed_type_importance', 'cancellation_policy_importance',
       'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
       'neighbourhood_mean_price']

#Andiamo a vedere la matrice di correlazione
sn.heatmap(df.corr(), annot=True)
plt.show()

#Notiamo che le feature ingegnerizzate "neighbourhood_mean_price" e "room_type_importance" correlano molto bene
#con il prezzo. In maniera minore anche la feature tv

#Vediamo che la feature "accomodates" correla 0.81 con "beds" quindi ne eliminamo una per evitare problemi di
#collinearità. Eliminiamo "beds" dato che "accomodates" correla maggiormente con il prezzo
df = df.drop("beds", axis=1)
#"accomodates" correla anche con "bedrooms" con 0.71 ma siccome si considera collineare sopra 0.7 la alsciamo

#Vediamo quindi che le feature che correlano di piu con il prezzo sono
#"room_type_importance" > "accomodates" > "neighbourhood_mean_price" > "bedrooms"

print(df.shape)

#Analizziamo room_type_importance con l'output in lineare
df["price"] = np.exp(df["log_price"])
sn.scatterplot(data=df, x="room_type_importance", y="price")
plt.show()

#Vediamo che ci sono valori per tutte e tre le categotorie: shared_room shared_house entire_house
#che sono a 0 cosa improbabile.
#Probabilmente sono outlier quindi andiamo ad eliminarli fissando una soglia minima di 10 dollari
print(len(df[df["price"] < 10])) #sono 2 row
df = df[df["price"] > 10]
sn.scatterplot(data=df, x="room_type_importance", y="price")
plt.show()

#Vediamo che ci sono valori per la shared room che vanno sopra i 600 dollari, cosa molto strana
#quindi andiamo a vedere quanti sono questi campioni
print(len(df[(df["price"] > 100) & (df["room_type_importance"] == 1)].index)) #189 campioni
#Infatti sopra 200 con shared room abbiamo solo 39 campioni che andiamo a droppare
df = df.drop(df[(df["price"] > 100) & (df["room_type_importance"] == 1)].index)
sn.scatterplot(data=df, x="room_type_importance", y="price")
plt.show()

#Contiamo ora i valore shared house sopra i 750 per importance 2
print(len(df[(df["price"] > 400) & (df["room_type_importance"] == 2)].index)) #212 campioni
#Abbiamo 67 campioni con sopra i 750 per una shared house, cosa molto improbabile quindi li droppiamo
df = df.drop(df[(df["price"] > 400) & (df["room_type_importance"] == 2)].index)

#Contiamo per entire house quanti valori sono sotto ai 20 dollari
print(len(df[(df["price"] < 50) & (df["room_type_importance"] == 3)].index)) #135
df = df.drop(df[(df["price"] < 50) & (df["room_type_importance"] == 3)].index)
print(len(df[(df["price"] > 1400) & (df["room_type_importance"] == 3)].index)) #168


#Vediamo il risultato
sn.scatterplot(data=df, x="room_type_importance", y="price")
plt.show()
#Cosi risulta piu ragionevole


#### Analizziamo accomodates con bedrooms ####
print("ACCOMODATES con BEDROOMS")
sn.scatterplot(data=df, x="accommodates", y="bedrooms")
plt.show()
#Qui vediamo che ci sono alcuni campioni con piu di 6 accomodates con 0 bedrooms
print(len(df[(df["accommodates"] > 6) & (df["bedrooms"] == 0)].index)) #54
df = df.drop(df[(df["accommodates"] > 6) & (df["bedrooms"] == 0)].index)

#### Analizziamo "bathrooms" con "bedrooms" ####
print("BATHROOMS con BEDROOMS")
sn.scatterplot(data=df, x="bathrooms", y="bedrooms")
plt.show()
#Vediamo che ci sono sample con 0 bedrooms ma con 1 o piu bagni cosa improbabile
print(len(df[(df["bathrooms"] > 1) & (df["bedrooms"] == 0)].index))#51
df = df.drop(df[(df["bathrooms"] > 1) & (df["bedrooms"] == 0)].index)

#Organizzo i dati per il training
#Vado a droppare anche le feature che correlano meno di 0.04 con l'uscita dato che non ci danno informazione
test = pd.read_csv("../test dataset nan feature eng/df_test_analisi.csv")
X_test = test.drop(labels=["log_price","price","distance_from_most_expensive","lux","parking","pool","number_of_reviews","instant_bookable"], axis=1)
y_test = test["log_price"]
X_train = df.drop(labels=["log_price","price","distance_from_most_expensive","lux","parking","pool","number_of_reviews","instant_bookable"], axis=1)
y_train = df["log_price"]

#vediamo la correlation matrix finale
df = df.drop(labels=["distance_from_most_expensive","lux","parking","pool","number_of_reviews","instant_bookable"], axis=1)
sn.heatmap(df.corr(), annot=True)
plt.show()

#le variabili categoriche le codifico con vettore one hot
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#Vado a normalizzare i dati. il testo verra normalizzato con la varianza e media del trainingset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Uso Linear Regression poichè le feature correlate con l'uscita sembrano avere un andamento lineare con l'uscita
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

#METRICHE SUL TRAINING SET
pred = regr.predict(X_train)
print("R2 training: %.2f" % r2_score(y_train, pred))
print("Mean squared error training: %.2f" % mean_squared_error(y_train, pred))

#METRICHE SUL TEST SET
pred = regr.predict(X_test)
print("R2 test: %.2f" % r2_score(y_test, pred))
print("Mean squared error test: %.2f" % mean_squared_error(y_test, pred))



parameters = {'silent': True,
              'learning_rate': 0.035,
              'iterations': 1900,
              'eval_metric': 'R2',
              'depth': 8,
              'allow_writing_files': False}
# R2: 0.7133 +- 0.0067


# parameters = {'silent': True,
#               'learning_rate': 0.023,
#               'iterations': 1005,
#               'eval_metric': 'R2',
#               'depth': 10,
#               'allow_writing_files': False}
# R2: 0.7061 +- 0.0078

import catboost as ctb

model_CBR = CatBoostRegressor(**parameters)

model_CBR.fit(X_train, y_train)

pred = model_CBR.predict(X_test)
print("R2 test: %.2f" % r2_score(y_test, pred))
print("Mean squared error test: %.2f" % mean_squared_error(y_test, pred))