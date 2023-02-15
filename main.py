import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

data = pd.read_csv("iris_data.csv")

data.rename({"variety": "Flower"}, axis="columns", inplace=True)

r_data = resample(data, replace=False)

x_data = r_data.drop("Flower", axis=1)
y_data = r_data["Flower"]

print("\t\t\t\t\t\t\t\tData\n\n" + str(data))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

scaled_x = scale(x_train)
scaled_x_test = scale(x_test)

svm = SVC()
svm.fit(scaled_x, y_train)

param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(svm, param_grid, refit=True, verbose=2)
grid.fit(scaled_x, y_train)
print("BEST ESTIMATOR\n" + str(grid.best_estimator_))

ConfusionMatrixDisplay.from_estimator(grid.best_estimator_, scaled_x_test, y_test)

pca = PCA()
PCA = pca.fit_transform(x_data)

per_var = pca.explained_variance_ratio_

plt.figure("Variance of PCA")
plt.bar(x=range(1, len(per_var)+1), height=per_var)

pca1 = PCA[:, 0]
pca2 = PCA[:, 1]
pca3 = PCA[:, 2]

x_min = pca1.min() - 1
x_max = pca1.max() + 1

y_min = pca2.min() - 1
y_max = pca2.max() + 1

z_min = pca3.min() - 1
z_max = pca3.max() + 1

new_svm_2 = SVC()

pca_2 = []
for i in range(len(pca1)):
    pca_2.append([])
    pca_2[i].append(pca1[i])
    pca_2[i].append(pca2[i])

new_svm_2.fit(pca_2, y_data)

param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(new_svm_2, param_grid, refit=True)
grid.fit(pca_2, y_data)

points = []
i = -5
j = -5
while i < 5:
    points.append([])
    points[len(points) - 1].append(i)
    points[len(points) - 1].append(j)
    j = j + 0.1
    if j > 5:
        j = -5
        i = i + 0.1

print("PCA BEST ESTIMATOR\n" + str(grid.best_params_))
pred = grid.best_estimator_.predict(points)
s = [[], []]
ve = [[], []]
vi = [[], []]

for i in range(len(points)):
    if pred[i] == "Setosa":
        s[0].append(points[i][0])
        s[1].append(points[i][1])
    elif pred[i] == "Versicolor":
        ve[0].append(points[i][0])
        ve[1].append(points[i][1])
    elif pred[i] == "Virginica":
        vi[0].append(points[i][0])
        vi[1].append(points[i][1])

plt.figure("My plot")
plt.scatter(ve[0], ve[1], alpha=0.3, label="Veronica predicted")
plt.scatter(vi[0], vi[1], alpha=0.3, label="Virginica predicted")
plt.scatter(s[0], s[1], alpha=0.3, label="Setosa predicted")

s = [[], []]
ve = [[], []]
vi = [[], []]
for i in range(len(y_data)):
    if y_data[y_data.index[i]] == "Setosa":
        s[0].append(pca1[i])
        s[1].append(pca2[i])
    elif y_data[y_data.index[i]] == "Versicolor":
        ve[0].append(pca1[i])
        ve[1].append(pca2[i])
    elif y_data[y_data.index[i]] == "Virginica":
        vi[0].append(pca1[i])
        vi[1].append(pca2[i])

plt.scatter(s[0], s[1], alpha=0.7, label="Veronica")
plt.scatter(ve[0], ve[1], alpha=0.7, label="Virginica")
plt.scatter(vi[0], vi[1], alpha=0.7, label="Setosa")
plt.legend()

xx, yy = np.meshgrid(np.round(np.arange(start=x_min, stop=x_max, step=0.01), decimals=2),
                     np.round(np.arange(start=y_min, stop=y_max, step=0.01), decimals=2))

Z = grid.best_estimator_.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
fig.canvas.set_window_title("Plot")

for i in range(len(Z)):
    for j in range(len(Z[i])):
        if Z[i][j] == "Setosa":
            Z[i][j] = 0
        elif Z[i][j] == "Virginica":
            Z[i][j] = 1
        else:
            Z[i][j] = 2
ax.contourf(xx, yy, Z, alpha=0.1)

plt.scatter(ve[0], ve[1], alpha=0.7, label="Veronica")
plt.scatter(vi[0], vi[1], alpha=0.7, label="Virginica")
plt.scatter(s[0], s[1], alpha=0.7, label="Setosa")


plt.legend()
plt.show()
