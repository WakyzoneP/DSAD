import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data_frame = pd.read_csv("data.csv", sep=";")
numeric_columns = data_frame.columns[2:].to_list()
for col in numeric_columns:
    data_frame[col].fillna(data_frame[col].mean(), inplace=True)

# print(numeric_columns)

# Calcul scoruri discriminante model liniar
X = data_frame[numeric_columns].drop(["Life_expectancy"], axis=1)
y = data_frame["Life_expectancy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
lda = LDA()
lda.fit(X_train_std, y_train)
lda_train_score = lda.score(X_train_std, y_train)
lda_test_score = lda.score(X_test_std, y_test)
print("LDA train score: ", lda_train_score)
print("LDA test score: ", lda_test_score)

# Trasare plot instanțe în axe discriminante
X_train_lda = lda.transform(X_train_std)
X_test_lda = lda.transform(X_test_std)
plt.scatter(X_train_lda[:,0], X_train_lda[:,1], c=y_train)
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Plot instanțe în axe discriminante")
plt.grid()
plt.savefig("plot_instante.png")

# Trasare plot distribuții în axele discriminante
plt.figure()
plt.hist(X_train_lda[y_train == 1, 0], label="1", bins=10, alpha=0.5)
plt.hist(X_train_lda[y_train == 2, 0], label="2", bins=10, alpha=0.5)
plt.hist(X_train_lda[y_train == 3, 0], label="3", bins=10, alpha=0.5)
plt.legend()
plt.xlabel("LD1")
plt.ylabel("Număr de instanțe")
plt.title("Plot distribuții în axele discriminante")
plt.grid()
plt.savefig("plot_distributii.png")

# Predicția în setul de testare model liniar
y_pred = lda.predict(X_test_std)
print("Predicted values: ", y_pred)

# Evaluare model liniar pe setul de testare (matricea de confuzie + indicatori de acuratețe)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc_score)

# Predicția în setul de aplicare model liniar ???????????
# X_apply = pd.read_csv("apply.csv", sep=";")
# X_apply_std = sc.transform(X_apply)
# y_apply = lda.predict(X_apply_std)

# Predicția în setul de testare model bayesian
min_samples_per_class = 2 
value_counts = y_train.value_counts()
to_keep = value_counts[value_counts >= min_samples_per_class].index
X_train_filtered = X_train_std[y_train.isin(to_keep)]
y_train_filtered = y_train[y_train.isin(to_keep)]

qda = QDA()
qda.fit(X_train_filtered, y_train_filtered)
qda_train_score = qda.score(X_train_filtered, y_train_filtered)
qda_test_score = qda.score(X_test_std, y_test)
print("QDA train score: ", qda_train_score)
print("QDA test score: ", qda_test_score)
y_pred = qda.predict(X_test_std)
print("Predicted values: ", y_pred)

# Evaluare model bayesian (matricea de confuzie + indicatori de acuratețe)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc_score)

# Predicția în setul de aplicare model bayesian ?????
# y_apply = qda.predict(X_apply_std)
# print("Predicted values: ", y_pred)