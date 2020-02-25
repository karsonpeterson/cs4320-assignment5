import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sys, os.path
import process_data
import collections

def display_histograms(fig_num, predictor_data, label_data, basename):
    column_names = process_data.get_all_feature_columns()
    label_column_names = process_data.get_label_columns()
    columns = predictor_data.shape[1]
    sp_rows = math.ceil(math.sqrt(columns) + 0.01)
    sp_cols = math.ceil(math.sqrt(columns))

    plt.suptitle('Feature histograms')
    fig = plt.figure(fig_num, figsize=(8.5, 11))
    for i in range(1, columns+1):
        name = column_names[i-1]
        plt.subplot(sp_rows, sp_cols, i)
        counts = collections.Counter(process_data.get_column(predictor_data, i-1))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel(name)

    plt.subplot(sp_rows, sp_cols, columns+1)
    counts = collections.Counter(process_data.get_column(label_data, 0))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel(label_column_names[0])
    plt.tight_layout()
    # plt.savefig(basename + '-histogram.pdf')
    plt.show()
    return

def display_slopes(fig_num, predictor_data, label_data, basename):
    column_names = process_data.get_all_feature_columns()
    label_column_names = process_data.get_label_columns();
    columns = predictor_data.shape[1]
    sp_rows = math.ceil(math.sqrt(columns) + 0.01)
    sp_cols = math.ceil(math.sqrt(columns))

    plt.suptitle('Labels vs Features')
    fig = plt.figure(fig_num, figsize(6.5, 9))
    Y = process_data.get_column(label_data, 0)
    for i in range(1, columns+1):
        name = column_names[i-1]
        plt.subplot(sp_rows, sp_cols, i)
        X = process_data.get_column(predictor_data, i-1)
        plt.scatter(X, Y, s=1)
        plt.xlabel(name)
        plt.ylabel(label_column_names[0])

    plt.subplot(sp_rows, sp_cols, columns+1)
    plt.scatter(Y, Y, s=1, color='blue')
    plt.xlabel(label_column_names[0])
    plt.ylabel(label_column_names[0])

    plt.tight_layout()
    # plt.savefig(basename + '-slopes.pdf')
    plt.show()

def display_data(predictor_data, label_data, basename):
    display_slopes(1, predictor_data, label_data, basename)
    display_histograms(2, predictor_data, label_data, basename)

def main(argv):
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = 'a.csv'

    if os.path.exists(filename):
        basename, ext = filename.split('.')
        data = process_data.get_data(filename)

        predictor_pipeline = process_data.make_predictor_pipeline(do_one_hot=False)
        label_pipeline = process_data.make_label_pipeline()

        predictors_processed = predictor_pipeline.fit_transform(data)
        labels_processed = label_pipeline.fit_transform(data)

        display_data(predictors_processed, labels_processed, basename)

    else:
        print(filename + " doesn't exist.")

    return

if __name__ == '__main__':
    main(sys.argv)