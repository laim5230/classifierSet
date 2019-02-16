#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from sklearn import metrics
import pickle
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn import tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model

# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier Grid Search
def random_forest_classifier_grid_search(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    # sklearn.model_selection.GridSearchCV(estimator,param_grid, scoring=None, fit_params=None, n_jobs=1,
    # iid=True, refit=True,cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise',return_train_score=True)
    param_grid = [
        {'bootstrap': [True],
         'n_estimators': [3, 10, 30],
         'max_features': [2, 4, 6, 8]
         },
        {'bootstrap': [False],
         'n_estimators': [3, 8, 10],
         'max_features': [2, 3, 4]
        }
    ]
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                   max_features=best_parameters['max_features'], bootstrap=best_parameters['bootstrap'])
    model.fit(train_x, train_y)
  #  feature_col = list(df.columns)
  #  sorted(zip(feature_importances, feature_col), reverse=True)
  #  final_model.fit(train_x, train_y)
    return model

# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(max_depth=6, criterion='entropy')
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier Grid Search
def decision_tree_classifier_grid_search(train_x, train_y):
    from sklearn import tree
    param_grid = [
        {'criterion': ['entropy', 'gini'],
         'max_depth': [4,5,7,9,15,20]
         }
    ]
    model = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = tree.DecisionTreeClassifier(criterion=best_parameters['criterion'],
                                        max_depth=best_parameters['max_depth'])
    model.fit(train_x, train_y)
    return model

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=60)
    model.fit(train_x, train_y)
    return model

# GBDT(Gradient Boosting Decision Tree) Classifier Grid Search
def gradient_boosting_classifier_grid_search(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    param_grid = [{'n_estimators': [30, 40, 50, 60, 80, 100]}]
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # grid_search.best_params_, grid_search.best_score_
    model = GradientBoostingClassifier(n_estimators=best_parameters['n_estimators'])
    model.fit(train_x, train_x)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM Classifier using cross validation
def svm_classifier_grid_search(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
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

def plot_matrix(test_y, predict):
    plt.imshow(metrics.confusion_matrix(test_y, predict), interpolation='nearest', cmap=plt.cm.binary)
    plt.grid(False)
    plt.colorbar()
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.show()

def load_data(data_file, short_col):
    my_data = pd.read_csv(data_file)
    if short_col != None:
        x = my_data[short_col]
        x = x.drop('label', axis=1)
    else:
        x = my_data.drop('label', axis=1)
    y = my_data['label']
    print("All data's shape is: {0}\n All data's columns are: {1}\n train_x's columns are: {2}\n"
          .format(my_data.shape, my_data.columns, x.columns))
    return x, y

def cv_metrics():
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics
    from sklearn import datasets
    from sklearn import svm
    clf = svm.SVC(kernel='linear', C=1)
    iris = datasets.load_iris()
    # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    scores = cross_val_score(clf, iris.data, iris.target, cv = 5, scoring = 'f1_macro')
    print(scores)
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    cross_val_score(clf, iris.data, iris.target, cv=cv)

def tree_graph(model, train_x, file_name):
    feature_names = train_x.columns
    # tree.export_graphviz(model, out_file="tree.dot13")
    import graphviz
    import pydotplus
    from IPython.display import Image
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_names,
                                    # class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True, proportion=True)
    # dot_data = tree.export_graphviz(model)
    # graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    graph = graphviz.Source(dot_data)
    graph.render(file_name)

def init_result_dict():
    auc_score = defaultdict(list)
    total_accuracy = defaultdict(list)
    model_set = defaultdict(list)
    return auc_score, total_accuracy, model_set

def print_model_presentation(test_y, predict):
    precision = metrics.precision_score(test_y, predict)
    recall = metrics.recall_score(test_y, predict)
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    confusion = metrics.confusion_matrix(test_y, predict)
    TP, TN, FP, FN = confusion[1, 1], confusion[0, 0], confusion[0, 1], confusion[1, 0]
    print('#TP : %d, TN: %d, #FP: %d, #FN: %d' % (TP, TN, FP, FN))
    acc_for_one, acc_for_zero = TP / (TP + FP), TN / (TN + FN)
    recall_for_one, recall_for_zero = TP / test_y.sum(), TN / (len(test_y) - test_y.sum())  # recall_for_one = TP/ (TP+FN)
    print('Accuracy of 0: %.2f%%, Recall of 0: %.2f%%' % (100 * acc_for_zero, 100 * recall_for_zero))
    print('Accuracy of 1: %.2f%%, Recall of 1: %.2f%%' % (100 * acc_for_one, 100 * recall_for_one))
    print('Total accuracy: %.2f%%' % (100 * accuracy))

def append_result_dict(test_y, predict, classifier, model):
    auc_score[classifier].append(metrics.roc_auc_score(test_y, predict))
    total_accuracy[classifier].append(100 * metrics.accuracy_score(test_y, predict))
    model_set[classifier].append(model)

"""
def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    my_unpickle = pickle._Unpickler(file=f, fix_imports=True,
                                   encoding="bytes", errors="strict")
    train, val, test = my_unpickle.load()
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y
"""
def print_tree_translator(model, train_x):
    from sk_tree import TreeTranslator
    inst = TreeTranslator(model.tree_, model.n_outputs_, model.classes_)
    print('--------------------- 75% for 1 rules: ----------------')
    inst.extract_rule(1, 0.75, train_x.columns)
    print('--------------------- 75% for 0 rules: ----------------')
    inst.extract_rule(0, 0.75, train_x.columns)
    print('--------------------- All rules: ----------------')
    inst.extract_rule(1, -1, train_x.columns)
    tree_graph(model, train_x, 'my_test1116')

def random_split(X, y, test_classifiers, test_size=.2, random_state=3234):
    print('------------ This is Random split --------------------------')
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    begin_do_model(test_classifiers, train_x, train_y, test_x, test_y)

def stratified_split(X, y, test_classifiers, n_splits, test_size, random_state):
    print('------------ This is Stratified split --------------------------')
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(X, y):
        train_x, test_x = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        begin_do_model(test_classifiers, train_x, train_y, test_x, test_y)

def begin_do_model(test_classifiers, train_x, train_y, test_x, test_y):
    num_train, num_feature = train_x.shape
    num_test, num_feature = test_x.shape
    print('******************** %s *********************' %time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feature))
    do_model(test_classifiers, train_x, train_y, test_x, test_y)

def do_model(test_classifiers, train_x, train_y, test_x, test_y):
        for classifier in test_classifiers:
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs' % (time.time() - start_time))
            predict = model.predict(test_x)
            if classifier == 'LR':
                print('LR coef is: ', model.coef_)
            if classifier == 'GBDT':
                print('GBDT importance is: ', sorted(zip(model.feature_importances_, train_x.columns)), reverse=True)
            if classifier == 'DT':
                # print_tree_translator(model, train_x)
                pass
            if model_save_file != None:
                model_save[classifier] = model
            is_binary_class = (len(np.unique(train_y)) == 2)
            if is_binary_class:
                print_model_presentation(test_y, predict)
                append_result_dict(test_y, predict, classifier, model)

if __name__ == '__main__':
    print(os.getcwd())
    # os.chdir('C:/Python27/charlie_project')
    thresh = 0.5
    model_save_file = None
    model_save = {}
    ## 'NB', 'LR', 'KNN', 'DT', 'RF', 'GBDT', 'SVM'
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'DT': decision_tree_classifier,
                   'DTGS': decision_tree_classifier_grid_search,
                   'RF': random_forest_classifier,
                   'RFGS': random_forest_classifier_grid_search,
                   'GBDT': gradient_boosting_classifier,
                   'GBDTGS': gradient_boosting_classifier_grid_search,
                   'SVM': svm_classifier,
                   'SVMGS': svm_classifier_grid_search
                   }

    data_file = 'new_short1117.csv'
    split_method = 'random'   # stratified
    test_classifiers = ['DT', 'NB']
    short_col_used = ['ip_tos', 'ip_len', 'ip_id', 'ip_off','ip_ttl', 'ip_sum', 'udp_dport','udp_len', 'udp_sum', 'payload_len','pay_entr_8_1', 'label']
    short_col_v2 = ['ip_tos', 'ip_len', 'ip_id', 'ip_off','ip_ttl', 'ip_sum', 'udp_sport', 'udp_dport','udp_len', 'udp_sum', 'order', 'label', 'payload_len','pay_entr_8_1']
    short_col = ['ip_tos', 'ip_len', 'ip_off', 'ip_ttl', 'udp_len', 'udp_sum']

    print('reading training and testing data...')
    X, y = load_data(data_file, short_col=None)
    # train_x, train_y = train_data.iloc[:,0:train_data.shape[1]-1], train_data['label']
    # test_x, test_y = test_data.iloc[:,0:test_data.shape[1]-1], test_data['label']
    auc_score, total_accuracy, model_set = init_result_dict()
    if split_method == 'stratified':
        stratified_split(X, y, n_splits=3, test_size=0.2, random_state=3234, test_classifiers=test_classifiers)
    elif split_method == 'random':
        random_split(X, y, test_classifiers=test_classifiers, test_size=0.2, random_state=3234)

    print("******************** All Model Effect Info *********************")
    print("AUC_socre: ", auc_score.items())
    print("total_accuracy: ", total_accuracy.items())
    # print("model_set: ", model_set.items())
    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
