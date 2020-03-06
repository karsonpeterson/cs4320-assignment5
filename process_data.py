import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.base
import sklearn.pipeline
import sklearn.impute
import scipy
import sys, os.path

def get_data(filename):
    data = pd.read_csv(filename)
    data['Seed1'] = data['Seed1'].str[1:3]
    data['Seed2'] = data['Seed2'].str[1:3]

    return data

def get_feature_columns():
    return ['Season', 'Seed1', 'Seed2', 'TeamID1', 'TeamID2']

def get_numerical_feature_columns():
    return ['Seed1', 'Seed2']

def get_categorical_feature_columns():
    return ['Season', 'TeamID1', 'TeamID2']

def get_all_feature_columns():
    return get_feature_columns()

def get_label_columns():
    return ['Result']

def get_column(data, i):
    if False:
        X = []
    elif isinstance(data, pd.core.frame.DataFrame):
        X = data.iloc[:,i]
    elif isinstance(data, np.ndarray):
        X = data[:,i]
    elif isinstance(data, scipy.sparse.csr.csr_matrix):
        X = data[:,i].todense()
    else:
        raise Exception('Data is unexpected type: ' + str(type(data)))

    return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Type:', type(X))
        print('Shape:', X.shape)
        print('X[0]', X[0])
        print(X)
        return X

class OutlierCuts(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        values = X
        # add cuts to data here
        return values

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, do_predictors=True, do_numerical=True):
        self.do_predictors = do_predictors
        self.do_numerical = do_numerical

        self.mCategoricalPredictors = get_categorical_feature_columns()
        self.mNumericalPredictors = get_numerical_feature_columns()
        self.mLabels = get_label_columns()

        return

    def fit(self, X, y=None):
        if self.do_predictors:
            if self.do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors
        else:
            self.mAttributes = self.mLabels
        
        return self

    def transform(self, X, y=None):
        values = X[self.mAttributes]
        return values

def make_numerical_predictor_params():
    params = {
        'features__numerical__numerical-predictors-only__do_predictors' : [True],
        'features__numerical__numerical-predictors-only__do_numerical' : [True],
        'features__numerical__missing-values__strategy' : ['median', 'mean', 'most_frequent']
    }
    return params

def make_categorical_predictor_params():
    params = {
        'features__categorical__categorical-predictors-only__do_predictors' : [True],
        'features__categorical__categorical-predictors-only__do_numerical' : [False],
        'features__categorical__missing-data__strategy' : ['most_frequent'],
        'features__categorical__encode-category-bits__categories': ['auto']
    }
    return params

def make_predictor_params():
    p1 = make_numerical_predictor_params()
    p2 = make_categorical_predictor_params()
    p1.update(p2)
    return p1

def make_numerical_predictor_pipeline():
    items = []
    items.append(('remove-outliers', OutlierCuts()))
    items.append(('numerical-predictors-only', DataFrameSelector(do_predictors=True, do_numerical=True)))
    items.append(('missing-values', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median')))
    items.append(('scaler', sklearn.preprocessing.StandardScaler(copy=False)))
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_predictor_pipeline(do_one_hot):
    items = []
    items.append(('remove-outliers', OutlierCuts()))
    items.append(('categorical-predictors-only', DataFrameSelector(do_predictors=True, do_numerical=False)))
    items.append(('missing-data', sklearn.impute.SimpleImputer(strategy='most_frequent')))
    if do_one_hot:
        items.append(('encode-category-bits', sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_predictor_pipeline(do_one_hot=False):
    items = []
    items.append(('numerical', make_numerical_predictor_pipeline()))
    items.append(('categorical', make_categorical_predictor_pipeline(do_one_hot)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def make_label_pipeline():
    items = []
    items.append(('labels-only', DataFrameSelector(do_predictors=False)))
    pipeline = sklearn.pipeline.Pipeline(items)
    return pipeline

def main(argv):
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'a.csv'

    if os.path.exists(filename):
        basename, ext = filename.split('.')
        data = get_data(filename)
        predictor_pipeline = make_predictor_pipeline(do_one_hot=True)
        label_pipeline = make_label_pipeline()
        predictors_processed = predictor_pipeline.fit_transform(data)
        labels_processed = label_pipeline.fit_transform(data)
        print(predictors_processed)
        print(labels_processed)
    else:
        print(filename + " doesn't exist.")

    return

if __name__ == '__main__':
    main(sys.argv)