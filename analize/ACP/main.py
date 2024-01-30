import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mortalitate_df = pd.read_csv('mortalitate_ro.csv', index_col=1)
# print(mortalitate_df.head(10))

numeric_columns = mortalitate_df.columns[1:].to_list()

for col in numeric_columns:
    mortalitate_df[col].fillna(mortalitate_df[col].mean(), inplace=True)
# print(mortalitate_df.head(10))
    
def standardize(data_frame):
    mean = np.mean(data_frame, axis=0)
    std = np.std(data_frame, axis=0)
    return (data_frame - mean) / std

X = mortalitate_df[numeric_columns].values
X_standardized = standardize(X)
data_frame_X_standardized = pd.DataFrame(data=X_standardized, index=mortalitate_df.index, columns=numeric_columns)

pca = PCA()
comp = pca.fit(X_standardized)

# Varianța Componentelor
variatie_componente = pca.explained_variance_ratio_

# Plot varianță componente cu evidențierea criteriilor de relevanță
plt.figure(figsize=(10, 8))
plt.plot(variatie_componente, 'ro-', linewidth=2)
plt.xlim(-1, 10)
plt.ylim(0, 1)
plt.title('Varianță Componente')
plt.xlabel('Componente')
plt.ylabel('Varianță')
plt.grid()
plt.savefig('./nou/variatie_componente.png')

# Calcul corelații factoriale (corelații variabile observate - componente)
corelatii_factoriale = np.corrcoef(X_standardized, comp.components_)
print(corelatii_factoriale)

# Trasare corelogramă corelații factoriale
plt.figure(figsize=(10, 8))
plt.imshow(corelatii_factoriale, cmap='coolwarm')
plt.colorbar()
plt.title('Corelogramă corelații factoriale')
plt.savefig('./nou/corelograma_corelatii_factoriale.png')

# Trasare cercul corelațiilor
plt.figure(figsize=(10, 10))
circle = plt.Circle((0, 0), 1, color='k', fill=False)
plt.gca().add_artist(circle)
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.title('Cercul corelațiilor')
plt.xlabel('Componenta 1')
plt.ylabel('Componenta 2')
plt.plot(corelatii_factoriale[0, :], corelatii_factoriale[1, :], 'ro')
for i in range(len(corelatii_factoriale[0, :])):
    # plt.annotate(numeric_columns[i], (corelatii_factoriale[0, i], corelatii_factoriale[1, i]))
    plt.arrow(0, 0, corelatii_factoriale[0, i], corelatii_factoriale[1, i], color='r', alpha=0.5)
plt.savefig('./nou/cercul_corelatiilor.png')

# Calcul componente și/sau scoruri
componente = pca.components_
componente_df = pd.DataFrame(data=componente, index=['C' + str(i) for i in range(1, comp.components_.shape[0] + 1)], columns=numeric_columns)
scoruri = pca.transform(X_standardized)

# Trasare plot componente
plt.figure(figsize=(10, 8))
plt.plot(componente[0, :], componente[1, :], 'ro')
plt.title('Plot componente')
plt.xlabel('Componenta 1')
plt.ylabel('Componenta 2')
plt.grid()
plt.savefig('./nou/plot_componente.png')

# Trasare plot scoruri
plt.figure(figsize=(10, 8))
plt.plot(scoruri[:, 0], scoruri[:, 1], 'ro')
plt.title('Plot scoruri')
plt.xlabel('Scor componenta 1')
plt.ylabel('Scor componenta 2')
plt.grid()
plt.savefig('./nou/plot_scoruri.png')

# Calcul cosinusuri
cosinusuri = np.square(componente) / pca.explained_variance_
print("Cosinusuri:", cosinusuri, sep='\n')

# Calcul contribuții
contributii = np.square(componente_df).div(np.sum(np.square(componente_df), axis=0), axis=1)
print("Contributii:", contributii, sep='\n')

# Calcul comunalități
comunalitati = np.sum(np.square(componente_df), axis=0)
print("Comunalitati:", comunalitati, sep='\n')

# Trasare corelogramă comunalități
plt.figure(figsize=(10, 8))
plt.imshow(np.corrcoef(X_standardized, comunalitati), cmap='coolwarm')
plt.colorbar()
plt.title('Corelogramă comunalități')
plt.savefig('./nou/corelograma_comunalitati.png')