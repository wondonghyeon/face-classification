from __future__ import print_function
import os
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str,
                        default='feature.csv',
                        help='path to the the feature file (default: feature.csv)')
    parser.add_argument('--label', type=str,
                        default='label.csv',
                        help='path to the label file (default: face_model.pkl)')
    parser.add_argument('--save_model', type=str,
                        default='face_model.pkl',
                        help='path to the model file to be pickled! (default: face_model.pkl)')

    args = parser.parse_args()

    # load features and labels
    print("reading data files from {}, and {}".format(args.feature, args.label))
    df_feat = pd.read_csv(args.feature, index_col=0)
    df_label = pd.read_csv(args.label, index_col=0)
    print("splitting train/test set (9:1)")
    # split training/test name
    unique_names = list(set([path.split('/')[0] for path in df_feat.index]))
    name_train, name_test = train_test_split(unique_names, test_size = 0.1, random_state = 0)
    name_train, name_test = set(name_train), set(name_test)
    # split training/test images
    idx_train = [path.split('/')[0] in name_train for path in df_feat.index]
    idx_test = [path.split('/')[0] in name_test for path in df_feat.index]
    X_train, Y_train = df_feat[idx_train], df_label[idx_train]
    X_test, Y_test = df_feat[idx_test], df_label[idx_test]
    print("start training MLP")
    # train models
    clf = MLPClassifier(solver='adam',
                        hidden_layer_sizes=(128, 128),
                        activation='relu',
                        max_iter = 5000,
                        verbose=True,
                        tol=1e-4)
    clf.fit(X_train, Y_train)
    print("saving the trained model to {}".format(args.save_model))
    with open(args.save_model, 'w') as f:
        pickle.dump([clf, df_label.columns.tolist()], f)


if __name__ == "__main__":
    main()
