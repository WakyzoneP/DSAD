from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def print_data(data):
    print(data)
    print("=====================================================================================================================================================================================")

# 1. Citirea corecta a datelor de intrare si afisarea acestora la consola. Pentru fiecare sursa de date se va crea un data-frame.
data_frame_agricultura = read_csv("Agricultura.csv")
data_frame_populatie_localitati = read_csv("PopulatieLocalitati.csv")

print_data(data_frame_agricultura)
print_data(data_frame_populatie_localitati)

# 2. Uniti cele 2 dataframe-uri create mai devreme. De asemenea verificati daca exista valori nule si curatati setul de date.
merged_data_frame = data_frame_agricultura.merge(data_frame_populatie_localitati, on="Siruta")
merged_data_frame_keys = merged_data_frame.keys()
for key in merged_data_frame_keys:
    if merged_data_frame[key].dtype == "float64":
        mean = merged_data_frame[key].mean()
        merged_data_frame[key].fillna(mean, inplace=True)
print_data(merged_data_frame)

# Să se salveze în fișierul Output1.csv valoarea totală a cifrei de afaceri (suma pentru activitățile menționate) la nivel de localitate. Pentru fiecare localitate se va salva codul Siruta, numele localității și cifra de afaceri.
data_frame_cifra_afaceri = DataFrame()
data_frame_cifra_afaceri['Siruta'] = merged_data_frame['Siruta']
data_frame_cifra_afaceri['Localitate'] = merged_data_frame['Localitate_x']
data_frame_cifra_afaceri['Cifra_afaceri'] = merged_data_frame['PlantePermanente'] + merged_data_frame['PlanteNepermanente'] + merged_data_frame['PlanteInmultire'] + merged_data_frame['CrestereaAnimalelor'] + merged_data_frame['FermeMixte'] + merged_data_frame['ActivitatiAuxiliare']
data_frame_cifra_afaceri.to_csv("Output1.csv", index=False)

#  Să se salveze în fișierul Output2.csv activitatea agricolă cu cifra de afaceri cea mai mare la nivel de localitate. Se va salva pentru fiecare localitate codul Siruta, denumirea localității și activitatea agricolă cu cea mai mare cifră de afaceri (este vorba de denumirea activității: PlanteNepermanente sau PlanteInmultire șamd)
data_frame_activitate = DataFrame()
data_frame_activitate['Siruta'] = merged_data_frame['Siruta']
data_frame_activitate['Localitate'] = merged_data_frame['Localitate_x']

def get_max_column(row):
    max_value = 0
    max_column = 'N/A'
    for column in data_frame_agricultura.columns:
        if data_frame_agricultura[column].dtype == "float64" and row[column] > max_value:
            max_value = row[column]
            max_column = column
    return max_column

data_frame_activitate['Activitate'] = data_frame_agricultura.apply(get_max_column, axis=1)
data_frame_activitate.to_csv("Output2.csv", index=False)

# Creati o clasa model pentru dataframe-ul vostru. Creati o lista de obiecte pe baza dataframe-ului.

class DataModel:
    def __init__(self, siruta, localitate, cifra_afaceri):
        self.siruta = siruta
        self.localitate = localitate
        self.cifra_afaceri = cifra_afaceri

data_model_objects = []

def create_objects(row):
    siruta = row['Siruta']
    localitate = row['Localitate']
    cifra_afaceri = row['Cifra_afaceri']
    data_model_object = DataModel(siruta, localitate, cifra_afaceri)
    data_model_objects.append(data_model_object)

data_frame_cifra_afaceri.apply(create_objects, axis=1)

def print_list(list, limit=None):
    if limit == None:
        for element in list:
            print(element.siruta, element.localitate, element.cifra_afaceri)
    else:
        for i in range(limit):
            print(list[i].siruta, list[i].localitate, list[i].cifra_afaceri)

print_list(data_model_objects, limit=10)

