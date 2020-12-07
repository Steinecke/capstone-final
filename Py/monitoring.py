#!/usr/bin/env python
"""
example performance monitoring script
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
from model import model_load
from sklearn.impute import SimpleImputer
from cslib import fetch_ts, engineer_features

def get_monitoring_tools(data_dir):
    ## remove the last 30 days (because the target is not reliable)
    ts_data = fetch_ts(data_dir)
    for country, df in ts_data.items():
        if country not in ['all', 'germany','united_kingdom']:
            continue
        y = df['y']
        X = df.drop(['y'], axis=1)
        n_samples = int(np.round(0.8 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples,replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        X = X[mask]
        y = y[mask]
        X.reset_index(drop=True, inplace=True)
        print("1)... X : ", X.shape, "         y:", y.shape)
        cols = list(X.select_dtypes([np.number]).columns)
        numeric_features = ['purchases', 'unique_invoices','unique_streams', 'total_views', 'revenue']
        print("... numeric features: ", numeric_features)
        X = X[cols]
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                          ('scaler', StandardScaler())])
        print("... numeric_transformer: ",numeric_transformer)
        transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
        print("... transformer: ", transformer)
        print("3)... X : ", X.shape, "         y:", y.shape)
        print("3)... X1 : ", X.shape, "         y:", y.shape)

        X_pp = transformer.fit_transform(X)
        print("... X_pp shape: ", X_pp.shape)

        xpipe = Pipeline(steps=[('pca', PCA(2)),('clf', EllipticEnvelope(random_state=0,contamination=0.01))])

        print("... xpipe:")

        xpipe.fit(X_pp)
        print("... xpipe.fit(X_pp):    ",xpipe.fit(X_pp))
        print("... Monitoring started ... ")
        samples = 10
        outliers_X = np.zeros(samples)
        wasserstein_X = np.zeros(samples)
        wasserstein_y = np.zeros(samples)
    
        for b in range(samples):
            n_samples = int(np.round(0.80 * X.shape[0]))
            subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=True).astype(int)
            y_bs=y[subset_indices]
#            print("y_bs", y_bs)
#           X_bs=X[subset_indices,:]
            X_bs=X_pp[subset_indices]
#            print("y_bs shape", y_bs.shape)
#            print("y shape", y.shape)
            XA=X.to_numpy()
            ya=y.to_numpy()
            yb=y_bs.to_numpy()
            print("y", y.head())
            print("ya", ya)
            print("y_bs", y_bs)
            test1 = xpipe.predict(X_bs)
            wasserstein_X[b] = wasserstein_distance(XA.flatten(),X_bs.flatten())
            wasserstein_y[b] = wasserstein_distance(ya.flatten(),yb.flatten())
            outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))
            outliers_X.sort()
            outlier_X_threshold = outliers_X[int(0.975*samples)] + outliers_X[int(0.025*samples)]
            wasserstein_X.sort()
            wasserstein_X_threshold = wasserstein_X[int(0.975*samples)] + wasserstein_X[int(0.025*samples)]
            wasserstein_y.sort()
            wasserstein_y_threshold = wasserstein_y[int(0.975*samples)] + wasserstein_y[int(0.025*samples)]
            to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_X":np.round(wasserstein_X_threshold,2),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "preprocessor":transformer,
                 "clf_X":xpipe,
                 "X_source":X,
                 "y_source":y}
            print("\n... Results .... \n", to_return)
        print("... monitoring finished ... ")
    return()

if __name__ == "__main__":

    print("... Start loading data.... ")
    data_dir = os.path.join(os.getcwd(),"data","cs-train")
    print("... End loading data.... ")
    print("... Start preprocessor ... ")
    get_monitoring_tools(data_dir)

