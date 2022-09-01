import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from more_itertools import flatten

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def ruff_model(train_df, cols, y, n=10, max_iter=10 ** 3, verbose=1, do_print=True):
    def get_encoder(col):
        if not isinstance(col, list):
            col = [col]

        if is_datetime(train_df[col[0]].dtype):
            return col[0].strip('_'+"".join(map(str,range(10)))), make_pipeline(FunctionTransformer(lambda x: x.apply(lambda y: y.dt.day_of_year)), StandardScaler()), col
        elif train_df[col[0]].dtype.kind == 'O':
            return col[0].strip('_'+"".join(map(str,range(10)))), OneHotEncoder(), col
        else:
            return col[0].strip('_'+"".join(map(str,range(10)))), StandardScaler(), col

    train_df = (train_df
                [list(flatten(cols)) + list(y)]
                .dropna()
                )

    X = train_df[flatten(cols)]
    y = train_df[y].to_numpy().astype('int').ravel()

    pipe = make_pipeline(ColumnTransformer(
        [get_encoder(col) for col in cols],
        remainder='drop'),
        SVC(max_iter=max_iter, shrinking=False, cache_size=2000))

    param_grid = {
        'svc__gamma': np.logspace(-5, 5, n),
        'svc__C': np.linspace(10 ** -2, 1, 5)
    }

    cv = GridSearchCV(pipe, param_grid=param_grid, verbose=verbose, n_jobs=-1)

    result = cv.fit(X, y)
    time.sleep(0.5)

    if do_print:
        print(result.best_score_, result.best_params_)
    return result


def my_predictors(df: pd.DataFrame, y):
    return pd.DataFrame({col: ruff_model(df, [col], y, verbose=False, do_print=False).best_score_
                         for col in df.columns if col not in y}, index=np.arange(1)).T.sort_values(0, ascending=False)
