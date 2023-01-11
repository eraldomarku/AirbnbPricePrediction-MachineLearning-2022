import numpy as np
import pandas as pd
import geopy.distance

# def fill_nan_neighbourhood(df, df2):
#     rimanente = 5400
#     for i,row in df.iterrows(): # A
#         a = row.latitude, row.longitude
#         distances = []
#         for j,row2 in df2.iterrows(): # B
#             b = row2.latitude, row2.longitude
#             distances.append(geopy.distance.geodesic(a, b).km)
#         min_distance = min(distances)
#         min_index = distances.index(min_distance)
#         df.iloc[i, df.columns.get_loc('neighbourhood')] = df2.iloc[min_index].loc["neighbourhood"]
#         #df.at[i, "neighbourhood"] = df2.iloc[min_index].loc["neighbourhood"]
#         #df.set_value(i, "neighbourhood", df2.iloc[min_index].loc["neighbourhood"].apply(str))
#         #df.iloc[i].loc["neighbourhood"] = df2.iloc[min_index].loc["neighbourhood"].apply(str)
#         #print(df.iloc[i].loc["neighbourhood"])
#         #print(df2.iloc[min_index].loc["neighbourhood"])
#         rimanente = rimanente - 1
#         print(rimanente)


def fill_nan_neighbourhood(df, df2, distanza_massima):
    rimanente = 5400
    neighbourhoods_vals = []
    for i,row in df.iterrows(): # A
        neighbourhood = None
        a = row.latitude, row.longitude
        for j,row2 in df2.iterrows(): # B
            if(row.city == row2.city):
                b = row2.latitude, row2.longitude
                distance = geopy.distance.geodesic(a, b).km
                if distance <= distanza_massima:
                    neighbourhood = row2.neighbourhood
                    break
            else:
                continue
        neighbourhoods_vals.append(neighbourhood)
        #df.at[i, "neighbourhood"] = str(neighbourhood)
        #df.set_value(i, "neighbourhood", df2.iloc[min_index].loc["neighbourhood"].apply(str))
        #df.iloc[i].loc["neighbourhood"] = neighbourhood
        #print(neighbourhood)
        #print(df.iloc[i].loc["neighbourhood"])
        #print(df2.iloc[min_index].loc["neighbourhood"])
        rimanente = rimanente - 1
        print(rimanente)
    df["neighbourhood"] = neighbourhoods_vals

def merge_dataframe_neighbourhood(df, df2):
    k = 0
    for i, row in df.iterrows():
        if pd.isnull(row.neighbourhood):
            for j, row2 in df2.iterrows():
                if(row["Unnamed: 0"] == row2["Unnamed: 0"]):
                        df.at[i, "neighbourhood"] = row2["neighbourhood"]
        k = k+1
        print(k)



def fill_nan_bedrooms(df, df2):
    rimanente = 53
    bedrooms_list = []
    for i,row in df.iterrows(): # A
        bedrooms = None
        for j,row2 in df2.iterrows(): # B
            if(row.city == row2.city):
                if row.accommodates == row2.accommodates:
                    bedrooms = row2.bedrooms
                    break
        bedrooms_list.append(bedrooms)
        print(bedrooms)
        rimanente = rimanente - 1
        print(rimanente)
    df["bedrooms"] = bedrooms_list



def merge_dataframe_bedrooms(df, df2):
    k = 0
    for i, row in df.iterrows():
        if pd.isnull(row.bedrooms):
            for j, row2 in df2.iterrows():
                if(row["Unnamed: 0"] == row2["Unnamed: 0"]):
                        df.at[i, "bedrooms"] = row2["bedrooms"]
        k = k+1
        print(k)



def fill_nan_beds(df, df2):
    rimanente = 161
    beds_list = []
    for i, row in df.iterrows():  # A
        beds = None
        for j, row2 in df2.iterrows():  # B
            if (row.city == row2.city):
                if row.accomodates == row2.accomodates:
                    beds = row2.beds
                    break
        beds_list.append(beds)
        print(beds)
        rimanente = rimanente - 1
        print(rimanente)
    df["bedrooms"] = beds_list

def fill_nan_bathrooms(df, df2):
    rimanente = 161
    bathrooms_list = []
    for i, row in df.iterrows():  # A
        bathrooms = None
        for j, row2 in df2.iterrows():  # B
            if (row.city == row2.city):
                if row.bedrooms == row2.bedrooms:
                    bathrooms = row2.bathrooms
                    break
        bathrooms_list.append(bathrooms)
        print(bathrooms)
        rimanente = rimanente - 1
        print(rimanente)
    df["bathrooms"] = bathrooms_list

def merge_dataframe_bathrooms(df, df2):
    k = 0
    for i, row in df.iterrows():
        if pd.isnull(row.bathrooms):
            for j, row2 in df2.iterrows():
                if(row["Unnamed: 0"] == row2["Unnamed: 0"]):
                        df.at[i, "bathrooms"] = row2["bathrooms"]
        k = k+1
        print(k)


import functools

from multiprocessing import Pool


def a(param1, param2, param3):
    return param1 + param2 + param3


def b(param1, param2):
    return param1 + param2


def smap(f):
    return f()

def jobss(df_nan1,df_nan2,df_nan3,df_nan4,df_nan5,df_nan6, df2):
    f_1 = fill_nan_neighbourhood(df_nan1, df2)
    f_2 = fill_nan_neighbourhood(df_nan2, df2)
    f_3 = fill_nan_neighbourhood(df_nan3, df2)
    f_4 = fill_nan_neighbourhood(df_nan4, df2)
    f_5 = fill_nan_neighbourhood(df_nan5, df2)
    f_6 = fill_nan_neighbourhood(df_nan6, df2)
    with Pool(6) as pool:
        res = pool.map(smap, [f_1, f_2, f_3, f_4, f_5, f_6])
        print(res)
