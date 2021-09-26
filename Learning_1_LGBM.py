import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
import gc
import random
import re
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import typing

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

def deviation_metric_realisation(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9

def deviation_metric(y_true: np.array, y_pred: np.array) -> float:
    return np.array([deviation_metric_realisation(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()

def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    return np.median(np.abs(y_pred-y_true)/y_true)

def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str,float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {'mape':mape, 'mdape':mdape, 'rmse': rmse, 'r2': r2, 'raif_metric':raif_metric}
train = pd.read_csv('C:/Users/lates/Desktop/drive-download-20210924T161153Z-001/data/train.csv', low_memory=False)
test = pd.read_csv('C:/Users/lates/Desktop/drive-download-20210924T161153Z-001/data/test.csv', low_memory=False)
test_submission = pd.read_csv('C:/Users/lates/Desktop/drive-download-20210924T161153Z-001/data/test_submission.csv', low_memory=False)
train = train[train['city'].isin(test['city'])]
train = train[train['price_type'] == 1].reset_index(drop=True)
train['date'] = train['date'].apply(lambda x: x.replace('-', '')).astype(int)
train['id'] = train['id'].apply(lambda x: x.replace('COL_', '')).astype(int)
test['date'] = test['date'].apply(lambda x: x.replace('-', '')).astype(int)
test['id'] = test['id'].apply(lambda x: x.replace('COL_', '')).astype(int)
target = train['per_square_meter_price'].values
train = train.drop('per_square_meter_price', axis=1)
data = pd.concat([train, test], axis=0).reset_index(drop=True)
data['floor'] = data['floor'].mask(data['floor'] == '-1.0', -1) \
              .mask(data['floor'] == '-2.0', -2) \
              .mask(data['floor'] == '-3.0', -3) \
              .mask(data['floor'] == 'подвал, 1', 1) \
              .mask(data['floor'] == 'подвал', -1) \
              .mask(data['floor'] == 'цоколь, 1', 1) \
              .mask(data['floor'] == '1,2,антресоль', 1) \
              .mask(data['floor'] == 'цоколь', 0) \
              .mask(data['floor'] == 'тех.этаж (6)', 6) \
              .mask(data['floor'] == 'Подвал', -1) \
              .mask(data['floor'] == 'Цоколь', 0) \
              .mask(data['floor'] == 'фактически на уровне 1 этажа', 1) \
              .mask(data['floor'] == '1,2,3', 1) \
              .mask(data['floor'] == '1, подвал', 1) \
              .mask(data['floor'] == '1,2,3,4', 1) \
              .mask(data['floor'] == '1,2', 1) \
              .mask(data['floor'] == '1,2,3,4,5', 1) \
              .mask(data['floor'] == '5, мансарда', 5) \
              .mask(data['floor'] == '1-й, подвал', 1) \
              .mask(data['floor'] == '1, подвал, антресоль', 1) \
              .mask(data['floor'] == 'мезонин', 2) \
              .mask(data['floor'] == 'подвал, 1-3', 1) \
              .mask(data['floor'] == '1 (Цокольный этаж)', 0) \
              .mask(data['floor'] == '3, Мансарда (4 эт)', 3) \
              .mask(data['floor'] == 'подвал,1', 1) \
              .mask(data['floor'] == '1, антресоль', 1) \
              .mask(data['floor'] == '1-3', 1) \
              .mask(data['floor'] == 'мансарда (4эт)', 4) \
              .mask(data['floor'] == '1, 2.', 1) \
              .mask(data['floor'] == 'подвал , 1 ', 1) \
              .mask(data['floor'] == '1, 2', 1) \
              .mask(data['floor'] == 'подвал, 1,2,3', 1) \
              .mask(data['floor'] == '1 + подвал (без отделки)', 1) \
              .mask(data['floor'] == 'мансарда', 3) \
              .mask(data['floor'] == '2,3', 2) \
              .mask(data['floor'] == '4, 5', 4) \
              .mask(data['floor'] == '1-й, 2-й', 1) \
              .mask(data['floor'] == '1 этаж, подвал', 1) \
              .mask(data['floor'] == '1, цоколь', 1) \
              .mask(data['floor'] == 'подвал, 1-7, техэтаж', 1) \
              .mask(data['floor'] == '3 (антресоль)', 3) \
              .mask(data['floor'] == '1, 2, 3', 1) \
              .mask(data['floor'] == 'Цоколь, 1,2(мансарда)', 1) \
              .mask(data['floor'] == 'подвал, 3. 4 этаж', 3) \
              .mask(data['floor'] == 'подвал, 1-4 этаж', 1) \
              .mask(data['floor'] == 'подва, 1.2 этаж', 1) \
              .mask(data['floor'] == '2, 3', 2) \
              .mask(data['floor'] == '7,8', 7) \
              .mask(data['floor'] == '1 этаж', 1) \
              .mask(data['floor'] == '1-й', 1) \
              .mask(data['floor'] == '3 этаж', 3) \
              .mask(data['floor'] == '4 этаж', 4) \
              .mask(data['floor'] == '5 этаж', 5) \
              .mask(data['floor'] == 'подвал,1,2,3,4,5', 1) \
              .mask(data['floor'] == 'подвал, цоколь, 1 этаж', 1) \
              .mask(data['floor'] == '3, мансарда', 3) \
              .mask(data['floor'] == 'цоколь, 1, 2,3,4,5,6', 1) \
              .mask(data['floor'] == ' 1, 2, Антресоль', 1) \
              .mask(data['floor'] == '3 этаж, мансарда (4 этаж)', 3) \
              .mask(data['floor'] == 'цокольный', 0) \
              .mask(data['floor'] == '1,2 ', 1) \
              .mask(data['floor'] == '3,4', 3) \
              .mask(data['floor'] == 'подвал, 1 и 4 этаж', 1) \
              .mask(data['floor'] == '5(мансарда)', 5) \
              .mask(data['floor'] == 'технический этаж,5,6', 5) \
              .mask(data['floor'] == ' 1-2, подвальный', 1) \
              .mask(data['floor'] == '1, 2, 3, мансардный', 1) \
              .mask(data['floor'] == 'подвал, 1, 2, 3', 1) \
              .mask(data['floor'] == '1,2,3, антресоль, технический этаж', 1) \
              .mask(data['floor'] == '3, 4', 3) \
              .mask(data['floor'] == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)', 1) \
              .mask(data['floor'] == '1,2,3,4, подвал', 1) \
              .mask(data['floor'] == '2-й', 2) \
              .mask(data['floor'] == '1, 2 этаж', 1) \
              .mask(data['floor'] == 'подвал, 1, 2', 1) \
              .mask(data['floor'] == '1-7', 1) \
              .mask(data['floor'] == '1 (по док-м цоколь)', 1) \
              .mask(data['floor'] == '1,2,подвал ', 1) \
              .mask(data['floor'] == 'подвал, 2', 2) \
              .mask(data['floor'] == 'подвал,1,2,3', 1) \
              .mask(data['floor'] == '1,2,3 этаж, подвал ', 1) \
              .mask(data['floor'] == '1,2,3 этаж, подвал', 1) \
              .mask(data['floor'] == '2, 3, 4, тех.этаж', 2) \
              .mask(data['floor'] == 'цокольный, 1,2', 1) \
              .mask(data['floor'] == 'Техническое подполье', -1) \
              .mask(data['floor'] == '1.2', 1) \
              .astype(float)
data = data.drop(['street'], axis=1)
categor_features = ['city', 'osm_city_nearest_name', 'region']

data['city'] = data['city'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
data['osm_city_nearest_name'] = data['osm_city_nearest_name'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))
data['region'] = data['region'].apply(lambda x: re.sub('[^A-ZА-Яа-яa-z0-9_]+', '', x))

for feat in categor_features:
    data_temp = pd.get_dummies(data[feat], drop_first=True)
    data.drop(feat, axis=1, inplace=True)
    data_temp.columns = [feat + '_' + str(col) for col in list(data_temp)]
    data = pd.concat([data, data_temp], axis=1)
    
train = data[data['id'].isin(train['id'])]
test = data[data['id'].isin(test['id'])]
target = target.copy()

import optuna
from optuna.samplers import TPESampler
sampler = TPESampler(seed=13)

def create_model(trial):
    num_leaves = trial.suggest_int("num_leaves", 2, 1500)
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth = trial.suggest_int('max_depth', 2, 25)
    min_child_samples = trial.suggest_int('min_child_samples', 2, 3000)
    learning_rate = trial.suggest_uniform('learning_rate', 0.00001, 0.99)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 2, 300)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.00001, 1.0)
    
    model = lgbm.LGBMRegressor(
        num_leaves=num_leaves,
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_child_samples=min_child_samples, 
        min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        random_state=13,
        n_jobs=-1
)
    return model

def objective(trial):
    model = create_model(trial)
    X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=random.randint(1, 10000))
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    score = deviation_metric(y_test.values, result)
    return score
params_lgbm = {'num_leaves': 887,
               'n_estimators': 480,
               'max_depth': 11,
               'min_child_samples': 1073,
               'learning_rate': 0.05348257149091985,
               'min_data_in_leaf': 2,
               'feature_fraction': 0.9529134909800754
              }
model = lgbm.LGBMRegressor(**params_lgbm)
%%time

kFold_random_state = [13, 666, 228, 777, 42]
n_splits = 10
final_loss = list()

submission = test_submission.copy()
submission.iloc[:, 1] = 0

for ind_k, random_state in enumerate(kFold_random_state):
    kFold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    total_loss = list()

    for iteration, (train_index, valid_index) in enumerate(kFold.split(train, target)):

        X_train, X_valid = train.iloc[train_index].reset_index(drop=True), train.iloc[valid_index].reset_index(drop=True)
        y_train, y_valid = target[train_index], target[valid_index]

        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        loss = deviation_metric(y_valid, valid_pred)

        predict = model.predict(test)
        submission['per_square_meter_price'] = submission['per_square_meter_price'] + predict / 50

        total_loss.append(np.mean(loss))

    final_loss.append(np.mean(total_loss))
    print(f'Fold({["1-10", "11-20", "21-30", "31-40", "41-50"][ind_k]}) deviation_metric: {np.mean(total_loss)}')
print(f'Final deviation_metric: {np.mean(final_loss)}')
submission_raw = submission.copy()
submission_raw['per_square_meter_price'] = submission_raw['per_square_meter_price'] * 0.9
submission_raw.loc[submission_raw['per_square_meter_price'] >= 200000, 'per_square_meter_price'] \
    = submission_raw.loc[submission_raw['per_square_meter_price'] >= 200000, 'per_square_meter_price'] * 0.9
submission_raw.to_csv('final_submission.csv', index=False)
