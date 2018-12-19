import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print(np.unique(iris_y))
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print("KNN:", knn.predict(iris_X_test))
print(iris_y_test)

svc_clf = SVC()
svc_clf.fit(iris_X_train, iris_y_train)
print("SVM:", svc_clf.predict(iris_X_test))
print(iris_y_test)


