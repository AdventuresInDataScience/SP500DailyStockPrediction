##############################################################################
# Data Config

# Paths
stocks_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/spy1d.csv"
constituents_path = "C:/Users/malha/Documents/Data/S&P 500 Historical Components & Changes(08-17-2024).csv"
upper_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/"
stocks_path_parquet = (
    "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/stocks1d.parquet.gzip"
)
final_data_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/final_data_parquet.gzip"
scaler_model_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/scaler_model_parquet.gzip"
ipca_model_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/ipca_model_parquet.gzip"

##############################################################################
# Model Config


# Model imports
from sklearn.linear_model import *
from lineartree import LinearForestRegressor
from gpboost import GPBoostRegressor
import KTBoost.KTBoost as KTBoost
import snapml.BoostingMachineRegressor as SnapboostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# Variables that need one hot encoding, and other paths
OHE_list = ["sector", "industry"]
scaler_model_path = (
    "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/scaler_model.joblib"
)
interim_optimisations_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/interim_optimisations.joblib"
pca_model_path = (
    "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/pca_model.joblib"
)
model_path = "C:/Users/malha/Documents/Data/SP500DailyStockPrediction/model.joblib"


# model list
ridge = RidgeCV()
rf = RandomForestRegressor(n_jobs=4)
lf = LinearForestRegressor(Ridge())
xgb = XGBRegressor()
lgbm = LGBMRegressor()
lgbml = lgb
gpb = GPBoostRegressor()
ktb = KTBoost.BoostingRegressor(loss="ls")
snap = SnapboostRegressor()

# model params lists
model_param_list = [
    (ridge, {"cv": [5, 10]}),
    (
        rf,
        {
            "max_depth": [10, 20, 30, None],
            "max_features": [1, "sqrt"],
            "min_samples_split": [2, 10, 25, 50],
            "min_samples_leaf": [1, 5, 10, 30],
        },
    ),
    (
        lf,
        {
            "max_depth": [5, 10, 20, 50, 100],
            "max_bins": [25, 50],
            "min_samples_split": [2, 6, 15, 25],
            "min_samples_leaf": [0.05, 0.1, 0.2],
        },
    ),
    (
        xgb,
        {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.3, 0.1, 0.01],
            "lambda": [1, 0.8, 0.5, 0.1],
            "alpha": [1, 0.8, 0.5, 0.1],
            "gamma": [0, 2, 4, 8],
            "min_child_weight": [0.3, 1, 4],
            "colsample_bytree": [0.5, 0.8, 1],
            "subsample": [0.5, 0.8, 1],
        },
    ),
    (
        lgbm,
        {
            "max_depth": [-1, 5, 10, 15],
            "learning_rate": [0.3, 0.1, 0.01],
            "colsample_bytree": [0.5, 0.8, 1],
            "subsample": [0.5, 0.8, 1],
            "reg_lambda": [1, 0.8, 0.5, 0.1, 0],
            "reg_alpha": [1, 0.8, 0.5, 0.1, 0],
            "n_estimators": [50, 100, 200],
        },
    ),
    (
        lgbml,
        {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 30,
            "learning_rate": 0.05,
            "verbosity": -1,
        },
    ),
    (
        gpb,
        {
            "boosting_type": ["gbdt", "dart", "goss", "rf"],
            "max_depth": [-1, 10, 20, 30],
            "learning_rate": [0.3, 0.1, 0.01],
            "n_estimators": [50, 100, 200],
            "colsample_bytree": [0.5, 0.8, 1],
            "reg_lambda": [1, 0.8, 0.5, 0.1, 0],
            "reg_alpha": [1, 0.8, 0.5, 0.1, 0],
        },
    ),
    (
        ktb,
        {
            "learning_rate": [0.3, 0.1, 0.01],
            "colsample_bytree": [0.5, 0.8, 1],
            "lambda_l2": [1, 0.8, 0.5, 0.1, 0],
            "alpha": [1, 0.8, 0.5, 0.1, 0],
        },
    ),
    (
        snap,
        {
            "base_learner": ["tree", "kernel"],
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.3, 0.1, 0.01],
            "max_depth": [3, 5, 10, 15],
            "gamma": [0.5, 0.7, 1],
            "kernel": ["rbf", "laplace", "GW"],
        },
    ),
]

"""
#Light GBM linear trees uses an very odd implementation. I need to add this last
I'll apply a single if statement check for light gbml, 
and if True, apply the additional linear data step'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Dataset for linear trees
train_data_linear = lgb.Dataset(X_train, label=y_train, params={'linear_tree': True})

# Dataset for regular trees
train_data_normal = lgb.Dataset(X_train, label=y_train)

# Specify model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 30,
    'learning_rate': 0.1,
    'verbosity': -1
}

# Train the models
model_linear = lgb.train(params, train_data_linear)
model_normal = lgb.train(params, train_data_normal)



#Early stopping rounds
present in XGBoost, LGBM, GPBoost, snapml, 
"""
##############################################################################


"""
random forest
ridge
XBnet

pip install mapie
https://github.com/scikit-learn-contrib/MAPIE

pip install --upgrade git+https://github.com/tusharsarkar3/XBNet.git
https://github.com/tusharsarkar3/XBNet/blob/master/XBNet/models.py
# Wheel build error - C++ 14

pip install --upgrade linear-tree
https://github.com/cerlymarco/linear-tree/blob/main/notebooks/README.md

pip install gpboost -U
https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPBoostRegressor.html#gpboost.GPBoostRegressor

pip install -U KTBoost
https://github.com/fabsig/KTBoost

#this is snapboost
pip install snapml
https://snapml.readthedocs.io/en/latest/boosting_machines.html#boosting-machine-regressor

pip install bartpy
https://github.com/JakeColtman/bartpy
#issue with deprecated sklearn vs scikit learn in requirements.txt

pip install SMAC3
pip install Cmake
pip install wheel setuptools --upgrade
# STILL NOT WORKING
https://github.com/automl/SMAC3

#alternative optimiser:
https://github.com/thuijskens/scikit-hyperband
git clone https://github.com/thuijskens/scikit-hyperband.git
python setup.py install



"""
