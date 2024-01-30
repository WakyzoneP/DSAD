import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2
import seaborn as sns

data_frame = pd.read_csv("DataSet_34.csv", index_col=0)
numeric_cols = data_frame.columns.tolist()
for column in numeric_cols:
    data_frame[column].fillna(data_frame[column].mean(), inplace=True)

X = data_frame[numeric_cols[:4]].values
Y = data_frame[numeric_cols[4:]].values

# Standardizare
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
Y_std = scaler.fit_transform(Y)

# Calcul scoruri canonice (variabile canonice)

n, p = X_std.shape
q = Y_std.shape[1]
m = min(p, q)
print(n, p, q, m)

cca = CCA(n_components=m)
cca.fit(X_std, Y_std)
x_c, y_c = cca.transform(X_std, Y_std)
print("x_c: ", x_c)
print("y_c: ", y_c)

# Calcul corelații canonice
cannonical_correlations = np.corrcoef(x_c.T, y_c.T).diagonal(offset=x_c.shape[1])
print("Corelații canonice: ", cannonical_correlations)


# Determinare relevanță rădăcini canonice (Test Bartlett)
def bartlett_test(cca, x_data, y_data):
    n = x_data.shape[0]
    p, q = x_data.shape[1], y_data.shape[1]
    # r = cca.n_components
    chi2_value = -((n - 1) - (p + q + 1) / 2) * np.sum(np.log(1 - cca.score(x_data, y_data) ** 2))
    df = p * q
    p_values = chi2.sf(chi2_value, df)
    return chi2_value, p_values


chi2_value, p_values = bartlett_test(cca, X_std, Y_std)
print("Chi2 value: ", chi2_value)
print("p value: ", p_values)

# Calcul corelații variabile observate - variabile canonice (Corelatii factoriale)
x_loadings = cca.x_loadings_
y_loadings = cca.y_loadings_
print("Corelații variabile observate - variabile canonice: ")
print("x_loadings: ", x_loadings)
print("y_loadings: ", y_loadings)

# Trasare plot corelații variabile observate - variabile canonice (cercul corelațiilor)
plt.figure(figsize=(8, 8))
cerc = plt.Circle((0, 0), radius=1, color='g', fill=False)
plt.gca().add_patch(cerc)
plt.axis('scaled')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('z1')
plt.ylabel('u1')
plt.grid()
plt.title('Cercul corelațiilor')
for i in range(m):
    plt.arrow(0, 0, x_loadings[i, 0], x_loadings[i, 1], color='r', alpha=0.5)
    plt.arrow(0, 0, y_loadings[i, 0], y_loadings[i, 1], color='b', alpha=0.5)
    plt.text(x_loadings[i, 0], x_loadings[i, 1], numeric_cols[i], color='r')
    plt.text(y_loadings[i, 0], y_loadings[i, 1], numeric_cols[i + 4], color='b')
plt.savefig('cercul_corelatiilor.png')

# Trasre corelograma corelații variabile observate - variabile canonice ???
df = pd.DataFrame(
    data=[x_loadings[:, 0], x_loadings[:, 1], y_loadings[:, 0], y_loadings[:, 1]],
    index=["z1", "z2", "u1", "u2"],
    columns=["x1", "x2", "y1", "y2"],
)
data = df.corr()
plt.figure(figsize=[10, 10])
sns.heatmap(data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corelograma corelațiilor")
plt.savefig("corelograma_corelatiilor.png")

# Trasare plot instanțe în spațiile celor două variabile (Biplot)
obs = data_frame.index.tolist()
plt.figure(figsize=(8, 8))
plt.scatter(x_c[:, 0], x_c[:, 1], marker='o', color='b', label='Set X')
plt.scatter(y_c[:, 0], y_c[:, 1], marker='x', color='r', label='Set Y')
plt.xlabel('X')
plt.ylabel('Y')
for i in range(len(obs)):
    plt.text(x_c[i, 0], x_c[i, 1], obs[i], color='b')
    plt.text(y_c[i, 0], y_c[i, 1], obs[i], color='r')
plt.grid()
plt.title('Biplot')
plt.legend(loc='best')
plt.savefig('biplot.png')

# Calcul varianță explicată și redundanță informațională
variance_explained = cannonical_correlations ** 2
print("Variance explained: ", variance_explained)
print("Information redundancy: ", 1 - variance_explained)