import pandas as pd
import numpy as np
import functii
import grafice

nume_fisier = "ADN/ADN_Tari.csv"

tabel = pd.read_csv(nume_fisier, index_col=0)
print(tabel,type(tabel),sep="\n")
variabile = list(tabel)
print(variabile)
variabile_selectate = variabile[1:]
# print(variabile_selectate)
nume_instante = list(tabel.index)
print(nume_instante)
x = tabel[variabile_selectate].values
print(x)

r,alpha,a,c,r_xc = functii.acp(x)
t_r = pd.DataFrame(data=r,
                   index=variabile_selectate,
                   columns=variabile_selectate)
t_r.to_csv("R.csv")

grafice.corelograma(t_r,"Corelograma corelatii variabile observate")

m = len(alpha)
etichete_componente = ["Comp"+str(i) for i in range(1,m+1)]
print(etichete_componente)


# Prezentare varianta
# Tabelare varianta
tabel_varianta = functii.tabelare_varianta(alpha,etichete_componente)
tabel_varianta.to_csv("Varianta.csv")
ncomp = grafice.plot_varianta(alpha)
grafice.show()
#
# # Prezentare corelatii variabile - componente
# t_r_xc = pd.DataFrame(data=r_xc,
#                    index=variabile_selectate,
#                    columns=etichete_componente)
# t_r_xc.to_csv("R_xc.csv")
# grafice.corelograma(t_r_xc,"Corelograma corelatii variabile observate - componente")
# for i in range(1,ncomp):
#     grafice.plot_variabile(r_xc,"Plot variabile",variabile_selectate,k2=i)
#
# # Prezentare scoruri
# # Calcul scoruri
# s = c / np.sqrt(alpha)
# pd.DataFrame(
#     data=s, index=nume_instante, columns=etichete_componente
# ).to_csv("Scoruri.csv")
# # Plot instante dupa componente si scoruri
# for i in range(1, ncomp):
#     grafice.plot_scoruri(c, "Plot instante (componente)", nume_instante, k2=i)
#     grafice.plot_scoruri(s, "Plot instante (scoruri)", nume_instante, k2=i)
#
#
# grafice.show()