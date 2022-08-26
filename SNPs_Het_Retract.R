
## Steps building pipeline
# 1.Create Environment for all ML tools
# 2.Import modules and add config file with default search parameters
# ADD: Prior input eval: Correlation matrix, values as tables, [train,test,val distribution]
#                        
# 3.Graphic updates on Tuning of fast models. namely:
#   a.Lasso
#   b.Ridge
#   c.LinearRegression
#   d.MLP
#   e.ElasticNet
# 4.Graphic updates on Tuning of heavy models. namely:
#   a.SVM
#   b.RandomForest
#   c.PolynomialRegression
#   Seperate for boosting models:
#   d.XGBoost
#   e.LGBoost
#   f.SGDBoost
# 5.Update tuned parameters in pickle file for final run
# 6.Scatter plot on final tuned parameters 


#Modules to run
from pyexpat.errors import XML_ERROR_PARTIAL_CHAR
import optuna
import h5py    
from numba import jit
import numpy as np    
from pyfiglet import Figlet
import pickle
import click
import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from time import time
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn  import metrics
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import sys
import warnings
import lightgbm as lgb
from mlxtend.regressor import StackingCVRegressor
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,recall_score,precision_score,log_loss,roc_auc_score,roc_curve
import yaml

#Functions req
def distribution_plot(data,title,method):
    title = 'Distribution of Subjects Age '+title
    fname = method+'/' +title
    ax = sns.histplot(data,kde=True).set(title=title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    return ax

def data_split(X,Y):
    # set aside 10% of train and test data for evaluation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
        test_size=0.1, shuffle = True, random_state = 42)
    # Use the same function above for the validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
        test_size=0.1/0.9, random_state= 42) 
    #Normalize data
    scaler = StandardScaler().fit(X_train)
    normalized_X_train = pd.DataFrame(
      scaler.transform(X_train),
      columns = X_train.columns
      )
    normalized_X_test = pd.DataFrame(
      scaler.transform(X_test),
      columns = X_test.columns
      )
    normalized_X_val = pd.DataFrame(
      scaler.transform(X_val),
      columns = X_test.columns
      )
    return normalized_X_train,normalized_X_test,normalized_X_val,Y_train,Y_test,Y_val


def all_metrics_to_file(test,pred_test,val,pred_val,method = 'ML',data = 'Whole Control',fileOut = 'MLResults.csv', save_metrics = True):
    mae = mean_absolute_error(test, pred_test)
    ev = explained_variance_score(test, pred_test)
    cor_p = pearsonr(test, pred_test)[0]
    r2_val = cor_p**2
    id_name = method + ' Test'
    print("\tTest Explained variance:", ev)
    print("\tTest Mean absolute error:", mae)
    print("\tTest Correlation:", cor_p)
    print("\tTest R2 score:", r2_val)
    print()
    res_df_test = pd.DataFrame(data = [method,mae,ev,cor_p,r2_val,data],index=['Method','Mean Absolute Error','Explained Variance','Correlation','R2 Score','Dataset Used'],columns = ['Test'])
    res_df_test = res_df_test.T
    mae_v = mean_absolute_error(val, pred_val)
    ev_v = explained_variance_score(val, pred_val)
    cor_p_v = pearsonr(val, pred_val)[0]
    r2_val_v = cor_p**2
    id_name = method + ' Validation'
    print("\tValidation Explained variance:", ev_v)
    print("\tValidation Mean absolute error:", mae_v)
    print("\tValidation Correlation:", cor_p_v)
    print("\tValidation R2 score:", r2_val_v)
    print()
    res_df_val = pd.DataFrame(data = [method,mae_v,ev_v,cor_p_v,r2_val_v,data],index=['Method','Mean Absolute Error','Explained Variance','Correlation','R2 Score','Dataset Used'],columns = ['Validation'])
    res_df_val = res_df_val.T
    final_res = pd.concat([res_df_test,res_df_val])
    if save_metrics:
        final_res.to_csv(fileOut, mode='a', header=not os.path.exists(fileOut))

def scat_plot(yt_pred,y_test,yv_pred,y_valid,title,method):
    title = 'Scatter Plot of Test Subjects Age '+title
    fname = method+'/Test_' +title
    data = pd.DataFrame(data = [list(yt_pred),list(y_test)], index=['Predicted','Test'])
    data = data.transpose()
    ax = sns.regplot(x='Predicted',y='Test',data = data).set(title=title,xlim = (35,80),ylim = (35,80))
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    title = 'Scatter Plot of Valid Subjects Age '+title
    fname = method+'/Valid_' +title
    data = pd.DataFrame(data = [list(yv_pred),list(y_valid)], index=['Predicted','Validation'])
    data = data.transpose()
    ax2 = sns.regplot(x='Predicted',y='Validation',data = data).set(title=title,xlim = (35,80),ylim = (35,80))
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()
    return ax,ax2

file_path_template = sys.argv[1]
file = open(file_path_template,'r')
cfc = yaml.load(file,Loader=yaml.FullLoader)


f = Figlet(font='slant')
print(f.renderText('ML_Autoptuna'))
print(" --------  Auto tuning on common ML methods in one stop with Optuna  --------")



# Uploading file
x_path =  str(cfc['input_files']['feature_path'])
y_path = str(cfc['input_files']['label_path'])

x = pd.read_csv(x_path,index_col = 'eid')
y = pd.read_csv(y_path,index_col = 'eid')

#Included for special case
y = y.loc[:,'Age_at_TestCenter']

#Can add what ratio to divide test,train and validation
normalized_X_train,normalized_X_test,normalized_X_val,Y_train,Y_test,Y_val = data_split(x,y)

#methods fasinclude lasso, ridge, 
fast_methods = str(cfc['methods']['fast'])
slow_methods = str(cfc['methods']['slow'])

#tuning params
tune_parameter = str(cfc['tuning']['parameter'])
tune_direction = str(cfc['tuning']['direction'])
tune_trials_fast = int(cfc['tuning']['num_trials_fast'])
tune_trials_slow = int(cfc['tuning']['num_trials_slow'])
lasso_alpha_range = cfc['params']['fast']['lasso_alpha']
lasso_tol_range = cfc['params']['fast']['lasso_tol']
elnet_alpha_range = cfc['params']['fast']['elnet_alpha']
elnet_ratio_range = cfc['params']['fast']['elnet_ratio']
ridge_alpha_range = cfc['params']['fast']['ridge_alpha']
ridge_tol_range = cfc['params']['fast']['ridge_tol']
hidden_layers_mlp = cfc['params']['fast']['hidden_layers_mlp'] #default (50,100),(100,100),(50,75,100),(25,50,75,100)
activation_mlp = cfc['params']['fast']['activation_mlp'] #default ["relu", "identity"]
learning_rate_init_mlp = cfc['params']['fast']['learning_rate_init_mlp']
solver_mlp = cfc['params']['fast']['solver_mlp']

def tune_fast(objective):
    study = optuna.create_study(direction=tune_direction)
    study.optimize(objective, n_trials=tune_trials_fast)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params

def tune_slow(objective):
    study = optuna.create_study(direction=tune_direction)
    study.optimize(objective, n_trials=tune_trials_slow)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params

def lasso_objective(trial):
    _alpha = trial.suggest_float("alpha", lasso_alpha_range[0], lasso_alpha_range[1])
    intercept = trial.suggest_categorical("fit_intercept", [True, False])
    tol = trial.suggest_float("tol",lasso_tol_range[0], lasso_tol_range[1], log=True)
    lasso = Lasso(alpha=_alpha, random_state=42, fit_intercept = intercept, max_iter=1000000, tol = tol)
    lasso.fit(normalized_X_train,Y_train)
    if tune_parameter == 'MAE':
        out = mean_absolute_error(Y_test, lasso.predict(normalized_X_test))
    elif tune_parameter == 'r2':
        out = r2_score(Y_test, lasso.predict(normalized_X_test))
    return out
    
def ridge_objective(trial):
    alpha = trial.suggest_float("alpha", ridge_alpha_range[0], ridge_alpha_range[1])
    intercept = trial.suggest_categorical("fit_intercept", [True, False])
    tol = trial.suggest_float("tol", ridge_tol_range[0], ridge_tol_range[1], log=True)
    regressor = Ridge(alpha=alpha,fit_intercept=intercept,tol=tol)
    regressor.fit(normalized_X_train, Y_train)
    if tune_parameter == 'MAE':
        out = mean_absolute_error(Y_test, regressor.predict(normalized_X_test))
    elif tune_parameter == 'r2':
        out = r2_score(Y_test, regressor.predict(normalized_X_test))
    return out

def mlp_regressor_objective(trial):
    hidden_layers = trial.suggest_categorical("hidden_layer_sizes", hidden_layers_mlp)
    activation = trial.suggest_categorical("activation",activation_mlp )
    if len(solver_mlp) == 1:
        solver = solver_mlp
    else:
        solver = trial.suggest_categorical("solver", solver_mlp)
    learning_rate = trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])
    learning_rate_init = trial.suggest_float("learning_rate_init", learning_rate_init_mlp[0], learning_rate_init_mlp[1])
    mlp_regressor = MLPRegressor(
                            hidden_layer_sizes=hidden_layers,
                            activation=activation,
                            solver=solver,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,
                            #early_stopping=True
                            )
    mlp_regressor.fit(normalized_X_train,Y_train)
    if tune_parameter == 'MAE':
        out = mean_absolute_error(Y_test, mlp_regressor.predict(normalized_X_test))
    elif tune_parameter == 'r2':
        out = r2_score(Y_test, mlp_regressor.predict(normalized_X_test))
    return out

def elnet_objective(trial):
    e_alphas = trial.suggest_float("alpha", elnet_alpha_range[0], elnet_alpha_range[1])
    intercept = trial.suggest_categorical("fit_intercept", [True, False])
    e_l1ratio = trial.suggest_float("tol", elnet_ratio_range[0], elnet_ratio_range[1], log=True)
    regressor = ElasticNet(max_iter=1e7, alpha=e_alphas, l1_ratio=e_l1ratio)
    regressor.fit(normalized_X_train, Y_train)
    if tune_parameter == 'MAE':
        out = mean_absolute_error(Y_test, regressor.predict(normalized_X_test))
    elif tune_parameter == 'r2':
        out = r2_score(Y_test, regressor.predict(normalized_X_test))
    return out

print("Running Linear Regression as baseline model...")
LR = linear_model.LinearRegression().fit(X=normalized_X_train, y=Y_train)

y_pred = LR.predict(normalized_X_test)
y_val = LR.predict(normalized_X_val)

distribution_plot(Y_train,'First Time point Training','InputDataDistribution')
distribution_plot(Y_test,'First Time point Testing','InputDataDistribution')
distribution_plot(Y_val,'First Time point Validation','InputDataDistribution')

distribution_plot(y_pred,'LinearRegression Test Distribution WC','LinearRegression')
distribution_plot(y_val,'LinearRegression Test Distribution WC','LinearRegression')

all_metrics_to_file(Y_test,y_pred,Y_val,y_val,method = 'Linear Regression',data='Whole Control')
scat_plot(Y_test,y_pred,Y_val,y_val,method = 'LinearRegression',title='Whole_Control')
print()

#Tuning the model for best params
if len(fast_methods) != 0:
    for met in fast_methods:
        if met == 'lasso':
            lasso_params = tune_fast(lasso_objective)
        elif met == 'ridge':
            ridge_params = tune_fast(ridge_objective)
        elif met == 'MLP':
            mlp_params = tune_fast(mlp_regressor_objective)
        elif met == 'elnet':
            elnet_params = tune_fast(elnet_objective)
    
#Running best params for results




