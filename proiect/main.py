import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

def print_function(title, content):
    separatorCount = 120 - len(title)
    separator = ""
    for i in range(separatorCount // 2):
        separator = separator + "="
    separator = separator + title
    for i in range(separatorCount // 2):
        separator = separator + "="
    print(separator)
    if content is not None:
        print(content)


# 1. Citirea corecta a datelor de intrare si afisarea acestora la consola
data_frame = pd.read_csv("data.csv", sep=";")
print_function("Datele de intrare", data_frame.head(10))
print_function("Informatii despre datele de intrare", data_frame.describe())
print_function("Tipuri de date", data_frame.dtypes)

# 2. Curatarea setului de date + standardizare
# am inlocuit datele lipsa de pe fiecare coloana cu media tuturor valorilor de pe acea coloana
years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
columns = ['GDP_per_capita', 'Unemployment', 'Public_educational_expenditure', 'Life_expectancy']
for year in years:
    year_row = data_frame.loc[data_frame['Year'] == year]
    for column in columns:
        year_row[column].fillna(year_row[column].mean(), inplace=True)
    
    data_frame.loc[data_frame['Year'] == year] = year_row
    

print_function('Datele de intrare dupa curatare', data_frame.head(10))

numeric_data = data_frame[columns]

scaler = StandardScaler()
date_scalate = scaler.fit_transform(numeric_data)
data_frame2 = pd.DataFrame(date_scalate, columns=columns)

data_frame_standard = pd.concat([data_frame2, data_frame.drop(columns=numeric_data)], axis=1)

print_function('Date Standardizate', data_frame_standard.head(10))

# 3. Problematica setului de date
# Am ales acest set de date pentru a analiza relatia 
# dintre caracteristicile proprietatilor si preturile acestora

# 4. Analiza datelor
# Vom folosi analiza de corelatie intre variabile

# Matricea de corelație arată în ce măsură două variabile numerice sunt asociate între ele.
# Valorile din matrice variază între -1 și 1. Un coeficient de corelație de 1 indică o corelație perfectă pozitivă
# (când o variabilă crește, și cealaltă crește), -1 indică o corelație perfectă negativă
# (când o variabilă crește, cealaltă scade), iar 0 indică lipsa unei corelații lineare.

data = data_frame[columns]
print_function('Date numerice', data.head())

matrice_corelatie = data.corr()

# corelatia dintre toate variabilele (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(matrice_corelatie, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corelație")
plt.show()
print_function('Matrice de corelatie', None)

# corelatia dintre dimensiunea casei si pret (scatterplot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Public_educational_expenditure', y='Unemployment', data = data_frame_standard)
plt.title("Corelatia dintre cheltuielile publice pentru educație și șomaj")
plt.xlabel("Cheltuielile publice pentru educație (Public_educational_expenditure)")
plt.ylabel("Șomaj (Unemployment)")
plt.show()

# corelatia dintre numarul de dormitoare si pret (barplot)
plt.figure(figsize=(8, 6))
sns.barplot(x='Public_educational_expenditure', y='GDP_per_capita', data = data_frame_standard)
plt.title("Corelatia dintre PIB-ul pe cap de locuitor și cheltuielile publice pentru educație")
plt.xlabel("Cheltuielile publice pentru educație (Public_educational_expenditure)")
plt.ylabel("PIB-ul pe cap de locuitor (GDP_per_capita)")
plt.show()

# Analiza canonica

# Analiza canonica (CCA) este o tehnica statistica utilizata pentru a identifica relatii liniare intre doua seturi de variabile.
# In codul furnizat, se efectueaza o analiza canonica intre variabilele din setul de date reprezentat de coloanele
# 'Unemployment', 'Public_educational_expenditure', 'Life_expectancy' si variabila 'GDP_per_capita'.

# Dupa standardizarea datelor, se aplica CCA pentru a obtine componente canonice (x_c si y_c)
# care maximizeaza corelatia intre cele doua seturi de variabile. 
# Se evalueaza apoi corelatia intre prima componenta canonica si variabila 'GDP_per_capita',
# iar rezultatele sunt vizualizate prin scatterplot-uri si boxplot-uri pentru a evidentia 
# relatii intre variabilele implicate in analiza canonica. 

x_data = data_frame[['Unemployment', 'Public_educational_expenditure', 'Life_expectancy']]
y_data = data_frame['GDP_per_capita']

x_canonic_matrix = (x_data - x_data.mean()) / x_data.std()
y_canonic_matrix = (y_data - y_data.mean()) / y_data.std()

cca = CCA(n_components=1)
cca.fit(x_canonic_matrix, y_canonic_matrix)
x_c, y_c = cca.transform(x_canonic_matrix, y_canonic_matrix)

cca_result = pd.DataFrame({'x_c': x_c[:, 0], 'y_c': y_c[:, 0], 'GDP_per_capita': data_frame['GDP_per_capita'], 'Unemployment': data_frame['Unemployment']})
print_function('Rezultatul analizei canonice', cca_result.head())

print_function('Corelatia dintre x_c si y_c', cca_result.corr())

sns.set_context("notebook", font_scale=1.1)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x_c', y='y_c', data=cca_result)
plt.title("Corelatia dintre x_c si y_c")
plt.xlabel("x_c")
plt.ylabel("y_c")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Unemployment', y='x_c', data=cca_result)
plt.title("Corelatia dintre șomaj și x_c")
plt.xlabel("Șomaj")
plt.ylabel("x_c")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x_c', y='y_c', hue='Unemployment', data=cca_result)
plt.title("Corelatia dintre x_c si y_c")
plt.xlabel("x_c")
plt.ylabel("y_c")
plt.show()

# Analiza PCA

# Analiza Componentelor Principale (PCA) este o tehnica de reducere a dimensionalitatii care transforma
# variabilele initiale intr-un set de componente ne-corelate, numite componente principale.
# Se selecteaza coloanele numerice, iar apoi datele sunt standardizate folosind StandardScaler.
# Aplicarea PCA genereaza componente principale care captureaza variatia datelor.

# Graficul rezultat reprezinta cele doua componente principale (PC1 si PC2) intr-un plan bidimensional,
# iar punctele sunt colorate in functie de pretul proprietatilor.
# Acest lucru ofera o perspectiva vizuala asupra distributiei datelor in produsul intern brut pe cap de locuitor.
# evidentiaza daca exista modele sau grupari in datele initiale. 
# Prin urmare, PCA permite o simplificare a datelor si furnizeaza o imagine sintetica
# a relatiilor dintre variabilele originale in contextul evolutiei produsului intern brut pe cap de locuitor.

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
primary_components = pca.fit_transform(data_scaled)

primary_data_frame = pd.DataFrame(data=primary_components[:, :2], columns=['PC1', 'PC2'])

target = data_frame['GDP_per_capita']
final_data_frame = pd.concat([primary_data_frame, target], axis=1)

plt.figure(figsize=(8, 6))
plt.scatter(final_data_frame['PC1'], final_data_frame['PC2'], c=final_data_frame['GDP_per_capita'], cmap='plasma')
plt.xlabel('Componenta principala 1')
plt.ylabel('Componenta principala 2')
plt.title("Analiza PCA")
plt.colorbar(label='PIB-ul pe cap de locuitor')
plt.show()

# 5. Concluzii

# Se poate observa că există o corelație pozitivă între cheltuielile publice pentru educație și PIB-ul pe cap de locuitor, 
# eea ce indică faptul că investițiile în educație pot avea un impact pozitiv asupra prosperității economice. 
# 
# În plus, analiza componentelor principale (PCA) și analiza canonică (CCA) oferă o perspectivă valoroasă asupra modului 
# în care diferite variabile sunt asociate cu PIB-ul pe cap de locuitor, evidențiind relații complexe și modele care ar 
# putea fi exploatate pentru planificare și politici economice. Pe de altă parte, corelația slabă între rata șomajului și 
# PIB poate indica faptul că există alți factori care influențează rata șomajului, independent de venitul per capita. 
# 
# Aceste descoperiri pot fi folosite pentru a ghida deciziile politice și a îmbunătăți planificarea economică.