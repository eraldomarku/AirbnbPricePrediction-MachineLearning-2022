import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import geopy.distance
import datetime
from datetime import datetime
from funzioni import fill_nan_bedrooms
from funzioni import merge_dataframe_bedrooms
from funzioni import fill_nan_bathrooms
from funzioni import merge_dataframe_bathrooms

#Carico il dataset
df = pd.read_csv('df_train.csv')


#Imposto visualizzate tutte le colonne
pd.options.display.max_columns = df.shape[1]

#Controllo il numero di nan: 28500
print(len(df) - len(df.dropna()))

#Vediamo il df
print(df.describe(include="all"))

#Togliamo le feature che non possono darci informazioni
df = df.drop(labels=["id", "thumbnail_url"], axis=1)

#Osserviamo i nan
print(df.isna().sum())
#Vediamo che i punti con 12-14k di nan sono: "first_review", "host_response_rate", "last_review", "review_scores_rating"
#Lasciamo intanto queste feature perchè le andremo a gestire dopo


##########################
# room_type
##########################
#Vediamo che ci sono 3 valori unique come riportato ai quali assegnamo un valore di importanza diverso
room_type_importance = {'Entire home/apt':3, 'Private room':2, 'Shared room':1}
df["room_type_importance"] = df["room_type"].map(room_type_importance)
#df = df.drop("room_type", axis=1)

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
# accomodates
##########################
#Questa rappresenta il numero di persone e va bene cosi


##########################
# bathrooms
##########################
#Ci sono 163 nan chee gestiremo in seguito



##########################
# bed_type
##########################
#Come vediamo ci sono 5 valori unique ai quali possiamo dare piu rilevanza degli altri
unique_bed_type = {'Real Bed':2,'Futon':2, 'Airbed':1, 'Couch':1, 'Pull-out Sofa':1}
df["bed_type_importance"] = df["bed_type"].map(unique_bed_type)
#df = df.drop("bed_type", axis=1)


##########################
# cancellation_policy
##########################
#Vediamo che gli unique sono 4 e sono ordinali quindi li converto
unique_cancellation_policy = {'flexible':1, 'moderate':2, 'strict':3, 'super_strict_30':4, 'super_strict_60':5}
#Ingegnerizziamo questa feature
df["cancellation_policy_importance"] = df["cancellation_policy"].map(unique_cancellation_policy)
#df = df.drop("cancellation_policy", axis=1)

################################
# cleaning_fee
################################
# Vediamo che questa colonna è gia bool


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
# first_review
################################
#Abbiamo detto che avremmo gestito in nan alla fine e cosi fare ma almeno adesso trasformo le date in ordinale
df["first_review_date"] = df["first_review"].apply(lambda x: datetime.strptime(str(x), '%Y-%M-%d').toordinal()
                                                        if (x == x) else None)
df = df.drop("first_review", axis=1)


################################
# host_has_profile_pic, host_identity_verified
################################
# Vediamo inoltre che ci sono 159 nan che gestiremo in seguito
#Vediamo che instant_bookable ha solo 2 unique ovvero t e f quindi lo convertiamo in type booleano
df["host_identity_verified"] = df["host_identity_verified"].map({"f":False, "t":True})
df["host_has_profile_pic"] = df["host_has_profile_pic"].map({"f":False, "t":True})



################################
# host_response_rate
################################
# Anche qua ci sono 14k nan che gestiremo in seguito
# vediamo che i valori unique sono tutte percentuali
print(df["host_response_rate"].unique())
# convertiamo queste percentuali in float
df["host_response_rate"] = df["host_response_rate"].str.rstrip('%').astype('float')


################################
# host_since
################################
#Anche qua abbiamo circa 159 nan che gestiremo in futuro
df["host_since_date"] = df["host_since"].apply(lambda x: datetime.strptime(str(x), '%Y-%M-%d').toordinal()
                                                        if (x == x) else None)
df = df.drop("host_since", axis=1)


################################
# instant_bookable
################################
df["instant_bookable"] = df["instant_bookable"].map({"f":False, "t":True})


################################
# last_review
################################
#Abbiamo detto che avremmo gestito in nan alla fine e cosi fare ma almeno adesso trasformo le date in ordinale
df["last_review_date"] = df["last_review"].apply(lambda x: datetime.strptime(str(x), '%Y-%M-%d').toordinal()
                                                        if (x == x) else None)
df = df.drop("last_review", axis=1)


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


################################
# name
################################
# Vediamo che ha 58k unique quindi saranno i nomi degli host quindi ha senso droppare questa colonna
df = df.drop("name", axis=1)


################################
# neighbourhood
################################
#Vediamo che abbiamo 611 unique e 5463 nan che gestiremo e li lasciamo cosi perchè potrebbe portarci informazione
#print(len(df["neighbourhood"].unique()))


################################
# number_of_reviews
################################
#La feature "number_of_reviews" ha minimo a 0 e massimo a 600 quindi sembrano valori ragionevoli


################################
# review_score_rating
################################
#La feature "review_score_rating" sembra avere valori ragionevoli ovvero min 20 e max 100
#Ci sono circa 13k nan che gestiremo in seguito

################################
# zipcode
################################
# Vediamo che zipcode contiene 756 unique quindi potrebbe essere un informazione importante
df['zipcode'] = df['zipcode'].apply(str)

################################
# bedrooms
################################
#La feature "bedrooms" ha minimo a 0 e massimo a 10 quindi sembrano valori ragionevoli


################################
# beds
################################
#La feature "beds" sembra avere il minimo a 0 il che è strano dato che su airbnb sono alloggi per dormire quindi eliminiamo questi campioni
df = df[df["beds"] > 0]



##################################################################################################
# Gestiamo i NaN
##################################################################################################
#Vediamo il df e i nan rimasti
print(df.describe(include="all"))
print(df.isna().sum())
print(df.keys())

##### last_review_date, "first_review_date", "last_review_date"#####
sn.heatmap(df[["host_since_date","first_review_date","last_review_date", "log_price"]].corr(), annot=True)
plt.show()
# Vediamo che first_review_date e last_review_date presentano circa 12k campioni nan e questi non sembrano correlare con
# L'uscita quindi è ragionevole eliminarsli
df = df.drop(labels=["host_since_date","first_review_date","last_review_date"], axis=1)

#### neighbourhood #####
#Vediamo che ci sono 5k nan
#Potremmo usare le coordinate per fillare questi valori
#Prendiamo solo le righe che hanno neighbourhood a nan
df_nan_neighbourhood = df[df["neighbourhood"].isna()]
#Prendiamo le righe che non hanno neighbourhood a nan
df_senza_nan_neighbourhood = df[df["neighbourhood"].notna()]
#df_1 = df_nan_neighbourhood.iloc[:908,:]
#df_2 = df_nan_neighbourhood.iloc[909:1818,:]
#df_3 = df_nan_neighbourhood.iloc[1819:2725,:]
#df_4 = df_nan_neighbourhood.iloc[2726:3633,:]
#df_5 = df_nan_neighbourhood.iloc[3633:4540,:]
#df_6 = df_nan_neighbourhood.iloc[4540:,:]
#jobss(df_1, df_2, df_3, df_4, df_5, df_6, df_senza_nan_neighbourhood)
# Andiamo a fillare i nan in questo modo:
# Per ogni riga nan scorro il df dei non nan controllando se appartengono alla stessa città
# Se è cosi vado a controllare se la distanza del non nan è <= a 1.5km
# Se è cosi allora prendo la neighbourhood del non nan e la sostituisco al nan
# E' stato implementato anche un metodo che calcola per ciascun nan la distanza con tutti i non nan e prende la distanza
# minima ma per motivi di tempo è stato utilizzato quest'ultimo
distanza_massima1 = 1.5
#fill_nan_neighbourhood(df_nan_neighbourhood, df_senza_nan_neighbourhood, distanza_massima1)
#df_nan_neighbourhood.to_csv("df_nan_neighbourhood.csv", index=False)
# Andiamo ora a vedere se df_nan_neighbourhood contiene ancora qualche nan
df_nan_neighbourhood = pd.read_csv('df_nan_neighbourhood.csv')
print(df_nan_neighbourhood.isna().sum())
# Vediamo che sono rimasti 788 nan quindi proviamo ad aumentare la distanza a 5km per riempirli
df_nan_neighbourhood2 = df_nan_neighbourhood[df_nan_neighbourhood["neighbourhood"].isna()]
distanza_massima2 = 5
#fill_nan_neighbourhood(df_nan_neighbourhood2, df_senza_nan_neighbourhood, distanza_massima2)
#df_nan_neighbourhood2.to_csv("df_nan_neighbourhood2.csv", index=False)
df_nan_neighbourhood2 = pd.read_csv('df_nan_neighbourhood2.csv')
# Andiamo a vedere quanti NaN sono rimasti
print(df_nan_neighbourhood2.isna().sum())
# Vediamo che sono rimasti 359 NaN e siccome neighbourhood potrebbe essere importante nella stima del prezzo perchè
# in base alla vicinanza ad alcuni posti poterbbe cambiare il prezzo quindi non posso andare a rifare la procedura con
# piu di 5km poichè rischio di assegnare neghbourhood importanti a campioni che non lo sono quindi droppo queste 359
# Prima di droppare però vado a ricostruire il dataset con i nuovi valori trovati
#merge_dataframe_neighbourhood(df_nan_neighbourhood, df_nan_neighbourhood2)
#merge_dataframe_neighbourhood(df, df_nan_neighbourhood)
#df.to_csv("df.csv", index=False)
df = pd.read_csv('df.csv')
print(df.isna().sum())
print(df.shape)
# Siccome ci sono rimasti 359 nan li andiamo a droppare per i motivi detti prima
df = df[df["neighbourhood"].notna()]
#Siccome neighbourhood sarebbero 600 feature dummizzate andiamo a creare una nuova feature che per ogni neighbourhood calcola il prezzo
#medio delle abitazioni
mean_price_neighbourhood_df = df.groupby('neighbourhood', as_index=False)['log_price'].mean()
mean_price_neighbourhood_dict = dict(zip(mean_price_neighbourhood_df['neighbourhood'], mean_price_neighbourhood_df['log_price']))
#Ora andiamo ad applicare la media dei prezzi alle varie neighbourhood creando una nuova feature
df["neighbourhood_mean_price"] = df["neighbourhood"].map(mean_price_neighbourhood_dict)

##### bedrooms #####
# Vediamo ora che ci sono 55 nan su bedrooms
print(df.isna().sum())
# Vediamo come correla con le altre feature
sn.heatmap(df.drop(["Unnamed: 0", "gym","air","latitude","longitude","parking","kitchen","pool"], axis=1).corr(), annot=True)
plt.show()
# Vediamo che "bedrooms" ragionevolmente correla 0.71 con "accomodates" quindi andremo a fillare questi nan con campioni della stessa città
# che hanno stesso numero di accomodates
df_nan_bedrooms = df[df["bedrooms"].isna()]
df_senza_nan_bedrooms = df[df["bedrooms"].notna()]
fill_nan_bedrooms(df_nan_bedrooms, df_senza_nan_bedrooms)
merge_dataframe_bedrooms(df, df_nan_bedrooms)
print(df.isna().sum())
print(df.shape)


##### zipcode #####
# Vediamo che zipcode ha 754 null ma siccome è quasi complementare di neighbourhood  possiamo eliminare la feature
# hanno stesso numero di unique ma ho scelto neighbourhood poichè lo zipcode potrebbe prendere sia una neighbourhood
# ricca che una povera
df = df.drop("zipcode", axis=1)

##### review_scores_rating #####
#Vediamo che presenta 13k campioni
sn.heatmap(df.drop(["Unnamed: 0", "gym","air","latitude","longitude","parking","kitchen","pool","instant_bookable"], axis=1).corr(), annot=True)
plt.show()
#Sembra non correlare per nulla con il prezzo quindi eliminiamo la feature siccome sono 13k campioni
df = df.drop("review_scores_rating", axis=1)

##### host_response_rate #####
#Vediamo che ci sono 14k nan
#Anche questa sembra non correlare per nulla con il prezzo quindi possiamo eliminare questa feature
df = df.drop("host_response_rate", axis=1)

##### host_identity_verified, host_has_profile_pic ####
#Abbiamo 159 nan ma sembra che anche queste non correlino con il prezzo quindi possiamo eliminare queste feature inoltre
#host_has_profile_pic ha quasi tutti true quindi non ci porta informazione
df = df.drop(labels=["host_identity_verified", "host_has_profile_pic"], axis=1)

##### bathrooms #####
#Vediamo ci sono 125 nan quindi siccome bathroom correla bene con la feature "bedrooms" quindi andiamo a fillare i nan con valore bathroom di
#altri sample della stessa citta che hanno stesso numero di bedrooms
df_nan_bathrooms = df[df["bathrooms"].isna()]
df_senza_nan_bathrooms = df[df["bathrooms"].notna()]
fill_nan_bathrooms(df_nan_bathrooms, df_senza_nan_bathrooms)
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

##### FINALE #####
# Andiamo a salvare in csv cosi da poter effettuare delle analisi in seguito
#df.to_csv("df_train_analisi.csv", index=False)

