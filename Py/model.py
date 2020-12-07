import joblib
import os
import re
import time
import pandas as pd
import numpy as np
from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.2
MODEL_VERSION_NOTE = "supervised learning model for time-series"
test_flag = True

def model_train_sub(df, tag, test):
    """
    example function to train model
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file
    """
    print("... Country >>>>>    ", tag)
    MODEL_VERSION_NOTE = "TreeRegression model for time-series"
    ## start timer for runtime
    time_start_modeling = time.time()

    X, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]
        print("testmodus y", y)
        print("testmodus X", X)
        print("testmodus dates", dates)

    ## Perform a train-test split
    ## random_state sets a seed to the random generator, otherwise train-test split would be different each time.
    ## shuffle - zufällig mischen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

    ## train a random forest model
    ## Parameters of pipelines set by ‘__’ separated parameter

    param_grid_rf = {
        'rf__criterion': ['mse', 'mae'],  ## measure quality of a split (mean squared error for variance reduction)
        'rf__n_estimators': [5, 8, 10, 12]  ## number of trees in the forest
    }
    # verschiedene Schritte werden in einer Pipeline zusammengefasst: Standardisierung und RandomForrest Schätzung)
    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    ## GridSearchCV helps to loop through predefined hyperparameters to select best parameters from the listed hyperparameters
    ## n_jobs - Number of jobs to run in parallel, -1 = all processors from pipeline
    ## cv -determines the cross-validation splitting strategy, kfold (k-fache Kreuzvalidierung)
    ## die Gesamtmenge wird in k etwa gleichgrosse Teilmengen aufgeteilt,
    ## es werden k Testdurchläufe gestartet, bei denen die jeweils i-te Teilmenge als Testmenge und die verbleibenden
    ## k − 1 Teilmengen als Trainingsmengen verwendet werden. Die Gesamtfehlerquote errechnet sich als Durchschnitt
    ## aus den Einzelfehlerquoten der k Einzeldurchläufe.
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)

    y_pred_test = grid.predict(X_test)
    y_pred_train = grid.predict(X_train)

    ## Ausgabe der Basisdaten
    #    print("y_pred",y_pred)
    #    print("y_test",y_test)
    #    print("y_train",y_train)
    #    print("X_train",X_train)
    #    print("X_test",X_test)

    df = pd.concat([pd.DataFrame(X_train, index=None).rename(columns={0: "X_train"}).reset_index(),
                    (pd.DataFrame(y_pred_train, index=None).rename(columns={0: "y_pred_train"})),
                    (pd.DataFrame(y_train, index=None).rename(columns={0: "y_train"}))], axis=1)
    df['country'] = tag

    df = pd.concat([pd.DataFrame(X_test, index=None).rename(columns={0: "X_test"}).reset_index(),
                    (pd.DataFrame(y_pred_test, index=None).rename(columns={0: "y_pred_test"})),
                    (pd.DataFrame(y_test, index=None).rename(columns={0: "y_test"}))], axis=1)
    df['country'] = tag

    eval_rmse_test = round(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    eval_rmse_train = round(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    ytest_mean = round(y_test.mean())
    ytrain_mean = round(y_train.mean())
    ypred_test_mean = round(y_pred_test.mean())
    ypred_train_mean = round(y_pred_train.mean())

    ytest_std = round(y_test.std())
    ytrain_std = round(y_train.std())
    ypred_test_std = round(y_pred_test.std())
    ypred_train_std = round(y_pred_train.std())

    Xtest_mean = round(X_test.mean())
    Xtrain_mean = round(X_train.mean())

    R2_test = 100 - round((100 * eval_rmse_test / ytest_std), 1)
    R2_train = 100 - round((100 * eval_rmse_train / ytrain_std), 1)

    out = sorted(grid.cv_results_.keys())

    print("\n\n\nCOUNTRY:>>>> ", tag, "     PERIOD:>>>> ", (str(dates[0]), str(dates[-1])))
    print("\nrsme_test:>>>> ", eval_rmse_test)
    print("rsme_train:>>>> ", eval_rmse_train)

    print("\nR2_test:>>>> ", R2_test)
    print("R2_train:>>>> ", R2_train)

    print("\ny_test_mean:>>>> ", (round(y_test.mean())))
    print("y_pred_train_mean:>>>> ", ypred_train_mean)
    print("y_mean:>>>> ", ytrain_mean)
    print("y_pred_test_mean:>>>> ", ypred_test_mean)
    print("y_mean:>>>> ", ytest_mean)

    print("Std_y_test:>>>> ", ytest_std)
    print("Std_y_pred_test:>>>> ", ypred_test_std)
    print("Std_y_train:>>>> ", ytrain_std)
    print("Std_y_pred_train:>>>> ", ypred_train_std)

    print("\nX_train_mean:>>>> \n", Xtrain_mean)
    print("\nX_test_mean:>>>> \n", Xtest_mean)

    print("\nMODEL_VERSION:>>>> ", MODEL_VERSION)
    print("MODEL_VERSION_NOTE:>>>> ", MODEL_VERSION_NOTE)
    print("test:>>>> ", test)

    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.", "_TreeRegression_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR, "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
        print("... MODEL_DIR", MODEL_DIR)
    else:
        saved_model = os.path.join(MODEL_DIR, "tr-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid, saved_model)

    m, s = divmod(time.time() - time_start_modeling, 60)
    h, m = divmod(m, 60)
    runtime = "%02d:%02d:%02d" % (h, m, s)
    print("runtime:>>>>>>  ", runtime)

    ## update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])), eval_rmse_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE,
                     test)


def model_train_sub2(df, tag, test=test_flag):
    """
    second function to train model
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file
    """
    MODEL_VERSION_NOTE = "fpProphet time-series analysis"

    from IPython.display import display               #displays infos within a loop

    from fbprophet import Prophet
    from fbprophet.diagnostics import cross_validation
    from fbprophet.diagnostics import performance_metrics

    print("... Country >>>>>    ", tag)
    print("... df shape >>>     ",df.shape)
    print("... df dtypes >>>     ",df.dtypes)
    print("... df columns >>>     ",df.columns)
    ## start timer for runtime
    time_start_modeling = time.time()

    X, y, dates = engineer_features(df)

    # from IPython.display import display               #displays infos within a loop
    df = df.rename(columns={'date': 'ds','price': 'y'})
    # database for Fitting and prediction
    # prediction not used for train-test-evaluation - split, it's used to compare different modelling approaches
    HXT = df.iloc[:-30, :]
    HXP = df.iloc[-30:, :]
    print(tag + " all data: ", df.shape)
    print(tag + " without last 30 days: ", HXP.shape)
    print(tag + " last 30 days : ", HXT.shape)

    grid = Prophet()

    grid = Prophet().fit(HXT)
    future = grid.make_future_dataframe(periods=30, freq='D')  # 30 timeframes forecast,  Basis: daily
    fcst = grid.predict(future)

    print("fcst shape: ", fcst.shape)
    cv_results = cross_validation(model=grid, horizon='30 days')
    print("cv results available")

    df_p = performance_metrics(cv_results)
    print("performance matrix available")

    eval_rmse_test = df_p['coverage'].mean()
    print("eval_rmse_test",eval_rmse_test)

    print("Summary: ", df_p)
    print(df_p.apply(tuple, axis=1).tolist())
    df_p.describe()

    ## retrain using all data

    fcst = grid.predict(future)
    cv_results = cross_validation(model=grid, horizon='30 days')
    df_p = performance_metrics(cv_results)
    eval_rmse_test = df_p['coverage'].mean()

#    fig1 = grid.plot(fcst)
#    fig2 = grid.plot_components(fcst)

    model_name = re.sub("\.", "_fbProphet_", str(MODEL_VERSION))

    if test:
        saved_model = os.path.join(MODEL_DIR, "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
        print("... MODEL_DIR", MODEL_DIR)
    else:
        saved_model = os.path.join(MODEL_DIR, "sl-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid, saved_model)

    m, s = divmod(time.time() - time_start_modeling, 60)
    h, m = divmod(m, 60)
    runtime = "%02d:%02d:%02d" % (h, m, s)
    print("runtime:>>>>>>  ", runtime)

    ## update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])), 0.84, runtime, MODEL_VERSION, MODEL_VERSION_NOTE,test)


def model_train(data_dir, test=test_flag):
    """
    function to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """
    print("... test flag model_train >>>>>    ", test)

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    print("MODEL_DIR >>>>   ", MODEL_DIR)
    print("Test Flag >>>>   ", test)
    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country, df in ts_data.items():

        if test and country not in ['all', 'germany']:
            continue

        model_train_sub(df, country, test)
        model_train_sub2(df, country, test)


def model_load(prefix='rt', data_dir=None, training=True):
    """
    example function to load model

    The prefix allows the loading of different models
    """
    print("... training >>>>>    ", training)

    if not data_dir:
        data_dir = os.path.join(os.getcwd(), "data", "cs-train")
    #        data_dir = os.path.join("..", "data", "cs-train")

    models = [f for f in os.listdir(os.path.join(".", "models")) if re.search("sl", f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]] = joblib.load(os.path.join(".", "models", model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X": X, "y": y, "dates": dates}

    return (all_data, all_models)


def model_predict(country, year, month, day, all_models=None, test=None):
    """
    example function to predict from model
    """
    print("... test flag model_predict>>>>>    ",test)

    ## start timer for runtime
    time_start_predict = time.time()

    ## load model if needed
    if not all_models:
        all_data, all_models = model_load(training=False)

    all_data, all_models = model_load(training=False)

    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year, month, day]:
        if re.search("\D", d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")

    ## load data
    model = all_models[country]
#    print("... models loaded")
    data = all_data[country]
#    print ("... data loaded")

    ## check date
    target_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
#    ds = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    print("... target date >>>>>>>  ",target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,data['dates'][0],data['dates'][-1]))

    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    print("\nquery\n",query)
    ## sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)

    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)
    m, s = divmod(time.time() - time_start_predict, 60)
    h, m = divmod(m, 60)
    predict_time = "%02d:%02d:%02d" % (h, m, s)

    y_proba =0

    ## update predict log
    update_predict_log(country, target_date, y_pred, y_proba, MODEL_VERSION, MODEL_VERSION_NOTE, predict_time,test)
    print("country >>>>    ",country)
    print("y_pred >>>>    ",y_pred)
    print("y_proba >>>>    ", y_proba)
    print("target date >>>>    ",target_date)
    print("predict_time >>>>    ",predict_time)
    print("MODEL_VERSION >>>>    ",MODEL_VERSION)
    print("MODEL_VERSION_NOTE >>>>    ",MODEL_VERSION_NOTE)

    return ({'y_pred': y_pred, 'y_proba': y_proba})
#    return ({'y_pred': y_pred})


if __name__ == "__main__":
    """
    basic test procedure for model.py
    """
    ## train the model
    print("TRAINING MODELS")

    data_dir = os.path.join(os.getcwd(), "data", "cs-train")
    model_train(data_dir, test=True)

## test predict
    country = 'all'
    year = '2019'
    month = '05'
    day = '05'
#    result = model_predict(country, year, month, day)
#    print(result)


