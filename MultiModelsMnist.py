import numpy as np
import time
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt
# pycharm中引用
matplotlib.use('Agg')

# 朴素贝叶斯
def native_bayes_classifier(train_x, train_y):
    # Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1)
    model.fit(train_x, train_y)
    return model


# KNN
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


#Logistic Regression Classifier
def logistic_regression_classifier(train_x,train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# 决策树
def decsion_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# 随机森林
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gbdt_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# 支持向量机
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def load_data():
    mnist = datasets.fetch_mldata('MNIST original')
    data, target = mnist.data, mnist.target
    print('data.shape:{},target.shape:{}'.format(data.shape, target.shape))
    index = np.random.choice(len(target), 70000, replace=False)
    # return data[index[:500]], target[index[:500]], data[index[500:600]], target[index[500:600]]
    return data[index[:60000]], target[index[:60000]], data[index[60000:]], target[index[60000:]]


if __name__ == '__main__':
    model_save_file = None
    model_save = {}
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'NB': native_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decsion_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gbdt_classifier
                   }
    print("加载数据....")
    train_x, train_y, test_x, test_y = load_data()

    # 保存训练监测数据
    train_time_cost = []
    test_time_cost = []
    model_accuracy = []

    # 画图
    fig, ax = plt.subplots()
    total_width, n = 0.8, 3
    width = total_width / n
    x = np.arange(7)

    for classifier in test_classifiers:
        print("******************** %s *************" % classifier )
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        train_time = time.time() - start_time
        print('training took %f s!' % train_time)
        train_time_cost.append(train_time)
        test_time_start = time.time()
        is_binary_class = (len(np.unique(train_y)) == 2)
        predict = model.predict(test_x)
        test_time = time.time() - start_time
        print('testing took %f s!' % test_time)
        test_time_cost.append(test_time)
        # 保存模型
        joblib.dump(model, './model/'+classifier+'.pkl')
        print(classification_report(test_y, predict))
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        model_accuracy.append(accuracy)
        print('accuracy: %.2f%%' % (100 * accuracy))

    ax.bar(x, model_accuracy, width=width, label="accuracy")
    ax.bar(x + width, train_time_cost, width=width, label="train_time")
    ax.bar(x + 2 * width, test_time_cost, width=width, label="test_time")
    ax.set_xticklabels(('NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT'))
    plt.savefig("./image/MultiModelsResult.png")
    plt.show()
