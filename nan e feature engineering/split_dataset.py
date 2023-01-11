import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')
df_train, df_test = train_test_split(df,test_size=0.2)
print(df_test.shape)
print(len(df_test) - len(df_test.dropna()))

df_train.to_csv("df_train.csv")
df_test.to_csv("df_test.csv")