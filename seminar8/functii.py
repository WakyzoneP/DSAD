import numpy as np
import pandas as pd

# Analiza in componente principale
def acp(x,std = True,nlib = 0):
    if std:
        # Standardizare
        x_ = (x - np.mean(x,axis=0))/np.std(x,axis=0)
    else:
        # Centrare
        x_ = x - np.mean(x, axis=0)
    n,m = np.shape(x)
    mat = (1/(n-nlib))*np.transpose(x_)@x_
    valp,vecp = np.linalg.eig(mat)
    # print(valp)
    # Sortare descrescatoare a valorilor proprii
    k = np.flipud(np.argsort(valp))
    # print(k)
    alpha = valp[k]
    a = vecp[:,k]
    # Calcul componente
    c = x_@a
    # Corelatii variabile - componente
    if std:
        r_xc = a*np.sqrt(alpha)
    else:
        r_xc = np.corrcoef(x_,c,rowvar=False)[:m,m:]
    return mat,alpha,a,c,r_xc

def tabelare_varianta(alpha,etichete_comp):
    procent = alpha*100/sum(alpha)
    tabel = pd.DataFrame(
        data = {
            "Varianta":alpha,
            "Varianta cumulata":np.cumsum(alpha),
            "Procent varianta":procent,
            "Procent cumulat":np.cumsum(procent)
        },
        index = etichete_comp
    )
    return tabel

