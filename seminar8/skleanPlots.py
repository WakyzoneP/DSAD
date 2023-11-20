import pandas as pd
import functii
import grafice
import numpy as np
import sklearn.decomposition as dec

nume_fisier = "ADN/ADN_Tari.csv"

tabel = pd.read_csv(nume_fisier, index_col=0)
# print(tabel,type(tabel),sep="\n")
variabile = list(tabel)
# print(variabile)
variabile_selectate = variabile[1:]
# print(variabile_selectate)
nume_instante = list(tabel.index)
# print(nume_instante)
x = tabel[variabile_selectate].values

# Construire model
model_acp = dec.PCA()
y = (x-np.mean(x,axis=0))/np.std(x,axis=0)
model_acp.fit(y)
c = model_acp.transform(y)
print(c[:, 0], c[:, 1]);
alpa = model_acp.explained_variance_
grafice.plot_varianta(alpa)
grafice.plot_scoruri(c,"Plot componente (sk)",nume_instante)
grafice.show()
