import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

data_frame = pd.read_csv("mortalitate_ro.csv", index_col=1)
numeric_columns = data_frame.columns[1:].to_list()
for col in numeric_columns:
    data_frame[col].fillna(data_frame[col].mean(), inplace=True)

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

X = data_frame[numeric_columns].values
data_std = standardize(X)

# Analiza factorabilității - Bartlett
chi_square_value, p_value = calculate_bartlett_sphericity(data_std)
print("Chi-square value: ", chi_square_value)
print("P-value: ", p_value)

# Analiza factorabilității - KMO
kmo_all, kmo_model = calculate_kmo(data_std)    
print("KMO value: ", kmo_model)
print("KMO all: ", kmo_all)

# Calcul varianță factori (fără rotație)
fa_without_rotation = FactorAnalyzer(n_factors=4, rotation=None)
fa_without_rotation.fit(data_std)
ev, v = fa_without_rotation.get_eigenvalues()
print("Eigenvalues: ", ev)
print("Variance: ", v)

# Calcul varianță factori (cu rotație)
fa_with_rotation= FactorAnalyzer(n_factors=4, rotation="varimax")
fa_with_rotation.fit(data_std)
ev, v = fa_with_rotation.get_eigenvalues()
print("Eigenvalues: ", ev)
print("Variance: ", v)

# Calcul corelații factoriale (fără rotație)
loadings = fa_without_rotation.loadings_
print("Loadings: ", loadings)

# Calcul corelații factoriale (cu rotație)
loadings = fa_with_rotation.loadings_
print("Loadings: ", loadings)

# Trasare corelogramă corelații factoriale (fără rotație)
loadings = fa_without_rotation.loadings_
plt.scatter(loadings[:,0], loadings[:,1])
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Corelogramă corelații factoriale")
plt.grid()
plt.savefig("corelograma_fara_rotatie.png")

# Trasare cercul corelațiilor (fără rotație)
loadings = fa_without_rotation.loadings_
plt.figure(figsize=(10,10))
plt.xlim(-1,1)
plt.ylim(-1,1)
circle = plt.Circle((0,0), radius=1, color="black", fill=False)
plt.gca().add_patch(circle)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Cercul corelațiilor")

for i in range(len(numeric_columns)):
    plt.arrow(0,0, loadings[i,0], loadings[i,1], color="red", alpha=0.5)
    plt.text(loadings[i,0]*1.2, loadings[i,1]*1.2, numeric_columns[i], color="black")
plt.grid()
plt.savefig("cercul_corelatiilor_fara_rotatie.png")

# Calcul comunalități și varianță specifică
comunality = fa_without_rotation.get_communalities()
specific_variance = fa_without_rotation.get_uniquenesses()
print("Comunality: ", comunality, sep="\n")
print("Specific variance: ", specific_variance, sep="\n")
# comunality = np.sum(np.square(loadings), axis=1)
# specific_variance = 1 - comunality
# print("Comunality: ", comunality, sep="\n")
# print("Specific variance: ", specific_variance, sep="\n")

# Trasare corelogramă comunalități și varianță specifică
plt.figure(figsize=(10, 10))
plt.bar(numeric_columns, comunality)
plt.xticks(rotation=90)
plt.savefig("comunalitati_fara_rotatie.png")
plt.clf()
plt.bar(numeric_columns, specific_variance)
plt.xticks(rotation=90)
plt.savefig("varianță_specifică_fara_rotatie.png")

# Calcul scoruri (fără rotație)
scores = fa_without_rotation.transform(data_std)
scores = pd.DataFrame(data=scores, index=data_frame.index, columns=['factor' + str(i) for i in range(1, 5)])
print("Scores: ", scores, sep="\n")

# Trasare plot scoruri
plt.figure(figsize=(10, 10))
plt.scatter(scores['factor1'], scores['factor2'])
plt.xlabel('factor1')
plt.ylabel('factor2')
for i in range(scores.shape[0]):
    plt.text(scores.iloc[i, 0], scores.iloc[i, 1], scores.index[i], fontsize=9)
plt.savefig("plot_scoruri_fara_rotatie.png")