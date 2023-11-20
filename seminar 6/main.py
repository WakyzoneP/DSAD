from pandas import *
from sklearn.preprocessing import *
from sklearn import *
from sklearn.datasets import load_iris

# 1. Creati un dataframe pornind de la fisierul incarcat.
data_frame = read_csv('Teritorial_2022.csv')
print(data_frame)
print("============================================================================================================================================================================================")

# 2. Afisati statistici descriptive despre data frame.
print(data_frame.describe())
print("============================================================================================================================================================================================")

# 3. Calculati nr mediu de medici per judet. 
medics_mean = data_frame['Medici'].mean()
print(medics_mean)
print("============================================================================================================================================================================================")

# 4. Inlocuiti valorile lipsa cu media. 
data_frame["Medici"].fillna(medics_mean, inplace=True)
print(data_frame)
print("============================================================================================================================================================================================")

# 5. Standardizati datele.
scaler = StandardScaler()
data = data_frame['Medici'].values.reshape(-1, 1)
scaled_data_frame = scaler.fit_transform(data)
print(scaled_data_frame)
print("============================================================================================================================================================================================")

iris_data_frame = DataFrame(load_iris().data, columns=load_iris().feature_names)
print(iris_data_frame)
scaled_data = scaler.fit_transform(iris_data_frame)
print(scaled_data)