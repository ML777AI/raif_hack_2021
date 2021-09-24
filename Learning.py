import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.style.use('classic')
%matplotlib inline

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OrdinalEncoder, FunctionTransformer, RobustScaler, Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from xgboost import cv
from xgboost import plot_importance
from sklearn.decomposition import PCA

#import tensorflow as tf
import matplotlib as mpl
import xgboost as xgb

pd.set_option('max_columns', None)

X_full = pd.read_csv("C:/Users/.../train.csv", index_col='id') # train
X_test_full = pd.read_csv("C:/Users/.../test.csv", index_col='id') # test

print(X_full)
print(X_test_full)

# Shape of training data (num_rows, num_columns)
print(X_full.shape)

# Shape of test data (num_rows, num_columns)
print(X_test_full.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_full.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['per_square_meter_price'], inplace=True)
y = X_full.per_square_meter_price
X_full.drop(['per_square_meter_price'], axis=1, inplace=True)

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

categorical_features  = [cname for cname in X_train.columns if
                    X_train[cname].nunique() <= 15 and 
                    X_train[cname].dtype == "object"]

numeric_features  = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']
                 ]

my_features = categorical_features + numeric_features
numeric_features.remove('per_square_meter_price')
print('numeric_features minus per_square_meter_price column:', numeric_features)

print(X_full.shape)
print(X_test_full.shape)
print()
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

  # get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]# Your code here

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

from sklearn.impute import SimpleImputer

# imputation
my_imputer = SimpleImputer()


imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# Preprocessed training and validation features
final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

#path='C:/Users/lates/Desktop/drive-download-20210924T161153Z-001/data'

output.to_csv('submission.csv', index=False)
