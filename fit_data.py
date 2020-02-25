import math
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.tree
import sklearn.svm
import sklearn.ensemble
import sys
import os.path
import getopt

import process_data

def make_decision_tree_params():
    params = process_data.make_predictor_params()
    tree_params = {
        'model__max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        'model__min_samples_split' : [0.001, 0.002, 0.003, 0.004, 0.005, 0.006], #0.007, 0.008, 0.009, 0.01, 0.05, 0.10, 0.20
        'model__criterion' : ['gini', 'entropy'],
        'model__splitter' : ['best', 'random'],
        'model__min_samples_leaf' : [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008], #0.009, 0.01, 0.05, 0.10, 0.20
        'model__max_leaf_nodes' : [2, 4, 8, 16, None],
        'model__min_impurity_decrease' : [0.000, 0.001, 0.002, 0.005, 0.010, 0.10]
    }
    params.update(tree_params)
    return params

def make_decision_tree_fit_pipeline():
    items = []
    items.append(('features', process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(('model', sklearn.tree.DecisionTreeClassifier()))
    return sklearn.pipeline.Pipeline(items)
    
def make_svm_params():
    params = process_data.make_predictor_params()
    svm_params = {
        'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'model__degree' : [1, 2, 3, 4, 5],
        'model__gamma' : ['auto', 'scale'],
        'model__coef0' : [-0.1, 0.0, 0.1]
    }
    params.update(svm_params)
    return params

def make_svm_fit_pipeline():
    items = []
    items.append(('features', process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(('model', sklearn.svm.SVC()))
    return sklearn.pipeline.Pipeline(items)

def make_bagging_tree_params():
    params = process_data.make_predictor_params()
    bagging_params = {
        'model__n_estimators' : [10, 20, 40, 80, 100, 120, 140],
        'model__max_samples' : [0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
        'model__max_features' : [0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
        'model__bootstrap' : [True, False],
        'model__bootstrap_features': [True, False]
    }
    params.update(bagging_params)
    return params

def make_bagging_tree_fit_pipeline():
    items = []
    items.append(('features', process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(('model', sklearn.ensemble.BaggingClassifier(sklearn.tree.DecisionTreeClassifier())))
    return sklearn.pipeline.Pipeline(items)


def make_adaboost_tree_params():
    params = process_data.make_predictor_params()
    adaboost_params = {
        'model__n_estimators' : [20, 40, 60, 80, 100, 120],
        'model__learning_rate' : [0.25, 0.5, 0.75, 0.8, 1.0]
    }
    params.update(adaboost_params)
    return params

def make_adaboost_tree_fit_pipeline():
    items = []
    items.append(('features', process_data.make_predictor_pipeline(do_one_hot=True)))
    items.append(('model', sklearn.ensemble.AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier())))
    return sklearn.pipeline.Pipeline(items)

def usage(short_opts, long_opts):
    print('usage: needs to be written')
    print(long_opts)
    return

def process_args(argv):
    allowed_model_types = ('tree', 'svm', 'bagging-tree', 'adaboost-tree')
    allowed_splitter_types = ('k-fold', 'stratified')
    allowed_search_types = ('grid', 'random')
    my_args = {
        'DataFileName' : 'data.csv',
        'ModelType' : 'tree',
        'SplitterType' : 'k-fold',
        'SearchType' : 'grid',
        'Folds' : 5,
        'Iterations' : 100
    }
    long_opts = ['help', 'file-name=', 'model=', 'splitter=', 'search=', 'folds=', 'iterations=']
    short_opts = 'hf:m:s:S:F:i:'
    try:
        opts, args = getopt.getopt(argv[1:], short_opts, long_opts)
    except getopt.GetoptError as e:
        print(e)
        usage(short_opts, long_opts)
        sys.exit(1)

    for o,a in opts:
        if o in ('-f', '--file-name'):
            my_args['DataFileName'] = a
        elif o in ('-m', '--model'):
            my_args['ModelType'] = a
        elif o in ('-s', '--splitter'):
            my_args['Splitter'] = a
        elif o in ('-S', '--search'):
            my_args['SearchType'] = a
        elif o in ('-F', '--folds'):
            my_args['Folds'] = int(a)
        elif o in ('-i', '--iterations'):
            my_args['Iterations'] = int(a)
        elif o in ('-h', '--help'):
            usage(short_opts, long_opts)
            sys.exit(1)
        else:
            usage(short_opts, long_opts)
            sys.exit(1)

    if not os.path.exists(my_args['DataFileName']):
        print()
        print('--file-name must be an existing file.')
        print(my_args['DataFileName'], 'does not exist.')
        print()
        usage(short_opts, long_opts)
        sys.exit(1)

    if my_args['ModelType'] not in allowed_model_types:
        print()
        print('--model must be one of: ', ' '.join(allowed_model_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)

    if my_args['SplitterType'] not in allowed_splitter_types:
        print()
        print('--splitter must be one of', ' '.join(allowed_splitter_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)

    if my_args['SearchType'] not in allowed_search_types:
        print()
        print('--search must be one of: ', ' '.join(allowed_search_types))
        print()
        usage(short_opts, long_opts)
        sys.exit(1)

    return my_args

def main(argv):

    my_args = process_args(argv)

    basename, ext = my_args['DataFileName'].split('.')
    data = process_data.get_data(my_args['DataFileName'])

    label_pipeline = process_data.make_label_pipeline()
    actual_labels = label_pipeline.fit_transform(data).values.ravel()

    if my_args['ModelType'] == 'tree':
        fit_pipeline = make_decision_tree_fit_pipeline()
        fit_params = make_decision_tree_params()

    elif my_args['ModelType'] == 'svm':
        fit_pipeline = make_svm_fit_pipeline()
        fit_params = make_svm_params()

    elif my_args['ModelType'] == 'bagging-tree':
        fit_pipeline = make_bagging_tree_fit_pipeline()
        fit_params = make_bagging_tree_params()

    elif my_args['Modeltype'] == 'adaboost-tree':
        fit_pipeline = make_adaboost_tree_fit_pipeline()
        fit_params = make_adaboost_tree_params()
    
    else:
        print('pick --model type')
        sys.exit(1)

    if my_args['SplitterType'] == 'k-fold':
        cv = sklearn.model_selection.KFold(n_splits=my_args['Folds'])

    elif my_args['SplitterType'] == 'stratified':
        cv = sklearn.model_selection.StratifiedKFold(n_splits=my_args['Folds'])

    else:
        print('pick --splitter type')
        sys.exit(1)

    if my_args['SearchType'] == 'grid':
        search_grid = sklearn.model_selection.GridSearchCV(fit_pipeline, fit_params, scoring='f1', n_jobs=-1,
        cv=cv, refit=True, verbose=1)

    elif my_args['SearchType'] == 'random':
        search_grid = sklearn.model_selection.RandomizedSearchCV(fit_pipeline, fit_params, scoring='f1', n_iter=my_args['Iterations'], n_jobs=-1, cv=cv, refit=True, verbose=1)

    else:
        print('pick --search type')
        sys.exit(1)

    search_grid.fit(data, actual_labels)

    print('Best Score: ', search_grid.best_score_)
    print('Best Params: ', search_grid.best_params_)

    print()
    print()
    print()

    predicted_labels = search_grid.best_estimator_.predict(data)

    # cm = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
    # print('Confusion Matrix:')
    # print('T')

    print()
    print('Precision: ', sklearn.metrics.precision_score(actual_labels, predicted_labels))
    print('Recall: ', sklearn.metrics.recall_score(actual_labels, predicted_labels))
    print('F1: ', sklearn.metrics.f1_score(actual_labels, predicted_labels))

    return


if __name__ == '__main__':
    main(sys.argv)