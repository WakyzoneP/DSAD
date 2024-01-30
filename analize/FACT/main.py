import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.decomposition import FactorAnalysis
from matplotlib.patches import Circle

data_frame = pd.read_csv("mortalitate_ro.csv", index_col=1)
numeric_variables = data_frame.columns[1:].to_list()
for col in numeric_variables:
    data_frame[col].fillna(data_frame[col].mean(), inplace=True)

print(data_frame.head())


def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

X = data_frame[numeric_variables].values
# print(X)
data_std = standardize(X)
# print(data_std)

def bartlett_test(data):
    corr_matrix = np.corrcoef(data, rowvar=False)
    n = data.shape[0]  # numărul de observații
    p = corr_matrix.shape[0]  # numărul de variabile
    chi_square = -((n - 1) - (2*p + 5)/6) * np.log(np.linalg.det(corr_matrix))
    p_value = chi2.sf(chi_square, p*(p-1)/2)
    return chi_square, p_value


chi_square_value, p_value = bartlett_test(data_std)
print("Chi-square value: ", chi_square_value)
print("P-value: ", p_value)

# Testul Kaiser-Meyer-Olkin
def kmo_test(data):
    corr_matrix = np.corrcoef(data, rowvar=False)
    partial_corr_matrix = np.linalg.inv(corr_matrix)
    kmo_num = np.sum(np.square(corr_matrix)) - np.sum(np.square(np.diagonal(partial_corr_matrix)))
    kmo_denom = kmo_num + np.sum(np.square(partial_corr_matrix)) - np.sum(np.square(np.diagonal(partial_corr_matrix)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value

kmo_value = kmo_test(data_std)
print("KMO value: ", kmo_value)

# Calcul varianță factori (cu/fără rotație)
fa = FactorAnalysis(n_components=4)  # Alege numărul de factori
fa.fit(data_std)
variance = fa.noise_variance_
print("Variance: ", variance)

# Calcul corelații factoriale (cu/fără rotație)
factor_loadings = fa.components_
factor_loadings = np.transpose(factor_loadings)
factor_loadings = pd.DataFrame(data=factor_loadings, index=numeric_variables, columns=['factor' + str(i) for i in range(1, 5)])
print("Factor loadings: ", factor_loadings)

# Trasare corelogramă corelații factoriale (cu/fără rotație)
plt.matshow(factor_loadings.corr())
plt.xticks(range(len(factor_loadings.columns)), factor_loadings.columns)
plt.yticks(range(len(factor_loadings.columns)), factor_loadings.columns)
plt.colorbar()
plt.savefig("corelograma.png")

# Trasare cercul corelațiilor (cu/fără rotație)
fig, ax = plt.subplots()
circle = Circle((0, 0), radius=1, edgecolor='black', facecolor='none')
ax.add_patch(circle)

for i in range(factor_loadings.shape[0]):
    x = factor_loadings.iloc[i, 0]
    y = factor_loadings.iloc[i, 1]
    plt.plot([0.0, x], [0.0, y], 'r')
    plt.text(x, y, factor_loadings.index[i], fontsize=9)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
plt.savefig("cercul_corelatiilor.png")

# Calcul comunalități și varianță specifică
comunality = np.sum(np.square(factor_loadings), axis=1)
specific_variance = 1 - comunality
print("Comunality: ", comunality, sep="\n")
print("Specific variance: ", specific_variance, sep="\n")

# Trasare corelogramă comunalități și varianță specifică
plt.figure(figsize=(10, 10))
plt.bar(numeric_variables, comunality)  
plt.xticks(rotation=90)
plt.savefig("comunalitati.png")
plt.clf()
plt.bar(numeric_variables, specific_variance)
plt.xticks(rotation=90)
plt.savefig("varianță_specifică.png")

# Calcul scoruri (cu/fără rotație)
scores = fa.transform(data_std)
scores = pd.DataFrame(data=scores, index=data_frame.index, columns=['factor' + str(i) for i in range(1, 5)])
print("Scores: ", scores, sep="\n")

# Trasare plot scoruri
plt.figure(figsize=(10, 10))
plt.scatter(scores['factor1'], scores['factor2'])
plt.xlabel('factor1')
plt.ylabel('factor2')
for i in range(scores.shape[0]):
    plt.text(scores.iloc[i, 0], scores.iloc[i, 1], scores.index[i], fontsize=9)
plt.savefig("plot_scoruri.png")
