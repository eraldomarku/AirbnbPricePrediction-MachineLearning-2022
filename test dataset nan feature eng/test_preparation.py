import numpy as np
import pandas as pd
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
from funzioni import fill_nan_neighbourhood
from funzioni import merge_dataframe_neighbourhood
from funzioni import fill_nan_bedrooms
from funzioni import merge_dataframe_bedrooms
from funzioni import fill_nan_bathrooms
from funzioni import merge_dataframe_bathrooms

feature_rimaste = ['log_price', 'property_type', 'accommodates', 'bathrooms',
       'cleaning_fee', 'city', 'instant_bookable', 'number_of_reviews',
       'bedrooms', 'beds', 'room_type_importance', 'pool', 'gym', 'air',
       'parking', 'kitchen', 'tv', 'number_of_amenities',
       'bed_type_importance', 'cancellation_policy_importance',
       'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
       'neighbourhood_mean_price']

#Carico il dataset
df = pd.read_csv('df_test.csv')

#Imposto visualizzate tutte le colonne
pd.options.display.max_columns = df.shape[1]

#Togliamo le feature che non possono darci informazioni
df = df.drop(labels=["id", "thumbnail_url"], axis=1)

#Osserviamo i nan
print(df.isna().sum())

#Eliminiamo feature che non useremo nel training
df = df.drop(labels=["beds","zipcode","review_scores_rating","name","last_review",
                     "host_since","host_identity_verified","host_has_profile_pic",
                     "first_review",], axis=1)

#Osserviamo i nan
print(df.isna().sum())

##########################
# room_type
##########################
#Vediamo che ci sono 3 valori unique come riportato ai quali assegnamo un valore di importanza diverso
room_type_importance = {'Entire home/apt':3, 'Private room':2, 'Shared room':1}
df["room_type_importance"] = df["room_type"].map(room_type_importance)

##########################
# amentities
##########################
#Amentities descrive i servizi offerti che possiamo ingegnerizzare creando feature chiave
df["pool"] = df["amenities"].str.lower().apply(lambda x: True if "pool" in x else False)
df["gym"] = df["amenities"].str.lower().apply(lambda x: True if "gym" in x else False)
df["air"] = df["amenities"].str.lower().apply(lambda x: True if "air" in x else False)
df["parking"] = df["amenities"].str.lower().apply(lambda x: True if "parking" in x else False)
df["kitchen"] = df["amenities"].str.lower().apply(lambda x: True if "kitchen" in x else False)
df["tv"] = df["amenities"].str.lower().apply(lambda x: True if "tv" in x else False)
#Creo una nuova fature basata sul numero di servizi
df["number_of_amenities"] = df["amenities"].str.lower().apply(lambda x: len(x.split(",")))
df = df.drop("amenities", axis=1)

##########################
# bed_type
##########################
#Come vediamo ci sono 5 valori unique ai quali possiamo dare piu rilevanza degli altri
unique_bed_type = {'Real Bed':2,'Futon':2, 'Airbed':1, 'Couch':1, 'Pull-out Sofa':1}
df["bed_type_importance"] = df["bed_type"].map(unique_bed_type)

##########################
# cancellation_policy
##########################
#Vediamo che gli unique sono 4 e sono ordinali quindi li converto
unique_cancellation_policy = {'flexible':1, 'moderate':2, 'strict':3, 'super_strict_30':4, 'super_strict_60':5}
#Ingegnerizziamo questa feature
df["cancellation_policy_importance"] = df["cancellation_policy"].map(unique_cancellation_policy)

################################
# city
################################
#Vediamo che city ha 6 unique che indicano la città quindi è ragionevole aggiungere
#un indice sul costo degli affitti in base alla città https://it.numbeo.com/
df["rent_cost_index"] = df["city"]
city_rent_cost = {'NYC':3394,'LA':2500, 'DC':2127, 'SF':2100, 'Boston':2600, 'Chicago':1900}
df["rent_cost_index"] = df["rent_cost_index"].map(city_rent_cost)


################################
# description
################################
#Questa feature descrive l'alloggio quindi anche da qua si possono ricavare delle feature chiave
#Vediamo che descrive il posto quindi possiamo aggiungere qualche feature il base a parole chiave da description
df["lux"] = df["description"].str.lower().apply(lambda x: True if "luxury" in x else False)
df["lux"] = df["description"].str.lower().apply(lambda x: True if "five star" in x else False)
df["lux"] = df["description"].str.lower().apply(lambda x: True if "5 star" in x else False)
df["view"] = df["description"].str.lower().apply(lambda x: True if "view" in x else False)
df = df.drop(labels=["description"], axis=1)

################################
# instant_bookable
################################
df["instant_bookable"] = df["instant_bookable"].map({"f":False, "t":True})

################################
# latitude, longitude
################################
#In questo modo ho trovato le coordinate per ciascuna città dell'abitazione piu costosa
most_expensive_house_lat = {'NYC':40.72823322373562, 'LA':34.10199071297228, 'DC':38.94701964733336, 'SF':37.78703531428325,
                   'Boston':42.34241482588717, 'Chicago':41.90528686674139}
most_expensive_house_long = {'NYC':-73.98933469704585, 'LA':-118.418380673005, 'DC':-77.06669920832232, 'SF':-122.49321543328428,
                    'Boston':-71.0745193431245, 'Chicago':-87.68346940732107}
#Nella feature "distance_from_most_expensive" calcola la distanza dall'abitazione all'abitazione piu costosa della sua città
df["most_expensive_house_lat"] = df["city"].map(most_expensive_house_lat)
df["most_expensive_house_long"] = df["city"].map(most_expensive_house_lat)
df["distance_from_most_expensive"] = df[["latitude", "longitude", "most_expensive_house_lat", "most_expensive_house_long"]].apply(
    lambda x: geopy.distance.geodesic((x["latitude"], x["longitude"]),(x["most_expensive_house_lat"],x["most_expensive_house_long"]),).km, axis=1)
#Elimino le feature ridondanti
df = df.drop(labels=["most_expensive_house_lat", "most_expensive_house_long"], axis=1)


print(df.isna().sum())

df_train = pd.read_csv('../nan e feature engineering/df.csv')

##################### GESTIAMO I NAN ################################################


#### neighbourhood #####
df_nan_neighbourhood = df[df["neighbourhood"].isna()]
distanza_massima1 = 1.5
#fill_nan_neighbourhood(df_nan_neighbourhood, df_train, distanza_massima1)
#df_nan_neighbourhood.to_csv("df_nan_neighbourhood.csv", index=False)
df_nan_neighbourhood = pd.read_csv('df_nan_neighbourhood.csv')
# Andiamo ora a vedere se df_nan_neighbourhood contiene ancora qualche nan
print(df_nan_neighbourhood.isna().sum())
# Vediamo che sono rimasti 82 nan quindi proviamo ad aumentare la distanza a 5km per riempirli
df_nan_neighbourhood2 = df_nan_neighbourhood[df_nan_neighbourhood["neighbourhood"].isna()]
distanza_massima2 = 5
#fill_nan_neighbourhood(df_nan_neighbourhood2, df_train, distanza_massima2)
#df_nan_neighbourhood2.to_csv("df_nan_neighbourhood2.csv", index=False)
df_nan_neighbourhood2 = pd.read_csv('df_nan_neighbourhood2.csv')
print(df_nan_neighbourhood2.isna().sum())
#Sono rimasti 72 neighbourhood nan che andremo a droppare
#Andiamo a fare il merge
#merge_dataframe_neighbourhood(df_nan_neighbourhood, df_nan_neighbourhood2)
#merge_dataframe_neighbourhood(df, df_nan_neighbourhood)
#df.to_csv("df.csv", index=False)
df = pd.read_csv('df.csv')
# Siccome ci sono rimasti 78 nan li andiamo a droppare per i motivi detti prima
df = df[df["neighbourhood"].notna()]
#Siccome neighbourhood sarebbero 600 feature dummizzate andiamo a creare una nuova feature che per ogni neighbourhood calcola il prezzo
#medio delle abitazioni
mean_price_neighbourhood_df = df_train.groupby('neighbourhood', as_index=False)['log_price'].mean()
mean_price_neighbourhood_dict = dict(zip(mean_price_neighbourhood_df['neighbourhood'], mean_price_neighbourhood_df['log_price']))
#Ora andiamo ad applicare la media dei prezzi alle varie neighbourhood creando una nuova feature
df["neighbourhood_mean_price"] = df["neighbourhood"].map(mean_price_neighbourhood_dict)
df = df[df["neighbourhood_mean_price"].notna()]

##### bedrooms #####
# Vediamo ora che ci sono 55 nan su bedrooms
print(df.isna().sum())
df_nan_bedrooms = df[df["bedrooms"].isna()]
fill_nan_bedrooms(df_nan_bedrooms, df_train)
merge_dataframe_bedrooms(df, df_nan_bedrooms)
print(df.isna().sum())
print(df.shape)

##### bathrooms #####
#Vediamo ci sono 37 nan
df_nan_bathrooms = df[df["bathrooms"].isna()]
fill_nan_bathrooms(df_nan_bathrooms, df_train)
merge_dataframe_bathrooms(df, df_nan_bathrooms)
print(df.isna().sum())
print(df.shape)

#### property_type ###
#Siccome 36 unique quindi li trasformiamo in: house, other, hotel, luxury
unique_property_type = {'Apartment':"house",
                        'House':"house",
                        'Other':"other",
                        'Condominium':"house",
                        'Bed & Breakfast':"hotel",
                        'Loft':"house",
                        'Townhouse':"hotel",
                        'Bungalow':"house",
                        'Earth House':"luxury",
                        'Villa':"house",
                        'Boat':'luxury',
                        'Guesthouse':"house",
                        'Cabin':"house",
                        'In-law':"house",
                        'Boutique hotel':"hotel",
                        'Timeshare':"house",
                        'Yurt':"house",
                        'Serviced apartment':"house",
                        'Tent':"other",
                        'Camper/RV':"other",
                        'Guest suite':"house",
                        'Dorm':"other",
                        'Hostel':"other",
                        'Castle':"luxury",
                        'Cave':"other",
                        'Hut':"other",
                        'Lighthouse':"luxury",
                        'Chalet':"house",
                        'Treehouse':"house",
                        'Train':"other",
                        'Tipi':"other",
                        'Casa particular':"house",
                        'Island':"luxury"}
df["property_type"] = df["property_type"].map(unique_property_type)


#Salviamo solo le feature che abbiamo nel training
df = df[['log_price', 'property_type', 'accommodates', 'bathrooms',
       'cleaning_fee', 'city', 'instant_bookable', 'number_of_reviews',
       'bedrooms','room_type_importance', 'pool', 'gym', 'air',
       'parking', 'kitchen', 'tv', 'number_of_amenities',
       'bed_type_importance', 'cancellation_policy_importance',
       'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
       'neighbourhood_mean_price']]

#Efettuo stesse operazioni fatte nel training
df["price"] = np.exp(df["log_price"])
df = df[df["price"] > 10]
df = df.drop(df[(df["price"] > 100) & (df["room_type_importance"] == 1)].index)
df = df.drop(df[(df["price"] > 400) & (df["room_type_importance"] == 2)].index)
df = df.drop(df[(df["price"] < 50) & (df["room_type_importance"] == 3)].index)
df = df.drop(df[(df["accommodates"] > 6) & (df["bedrooms"] == 0)].index)
df = df.drop(df[(df["bathrooms"] > 1) & (df["bedrooms"] == 0)].index)


#df.to_csv("df_test_analisi.csv", index=False)
print(df.isna().sum())















feature_rimaste = ['log_price', 'property_type', 'accommodates', 'bathrooms',
       'cleaning_fee', 'city', 'instant_bookable', 'number_of_reviews',
       'bedrooms', 'beds', 'room_type_importance', 'pool', 'gym', 'air',
       'parking', 'kitchen', 'tv', 'number_of_amenities',
       'bed_type_importance', 'cancellation_policy_importance',
       'rent_cost_index', 'lux', 'view', 'distance_from_most_expensive',
       'neighbourhood_mean_price']



