#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:26:04 2017

@author: LinZhang
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore') #Supress unnecessary warnings for readability
from scipy import stats
from scipy.stats import norm, skew, boxcox #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
from scipy.special import boxcox1p

#####################
# Data Loading
#####################
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
#####################
# Data Processing
#####################
######
# Outlier: according to the document: http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt
######
#==============================================================================
# #Show the outliers
# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()
# plt.close()
#==============================================================================
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
######
# Target Variable: reshape to Gaussian
######
#==============================================================================
# sns.distplot(train['SalePrice'] , fit=norm);
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# 
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
# plt.close()
#==============================================================================
# The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

# reshape the dist with log(1+x)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])
#==============================================================================
# #Check the new distribution 
# sns.distplot(train['SalePrice'] , fit=norm)
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# 
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
# plt.close()
#==============================================================================
######
# Missing values
######
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
# missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
#Plot the missing columns with percentage
#==============================================================================
# f, ax = plt.subplots(figsize=(15, 12))
# plt.xticks(rotation='90')
# sns.barplot(x=all_data_na.index, y=all_data_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by feature', fontsize=15)
# plt.close()
#==============================================================================
######
# Data Correlation
######
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.close()
######
# Imputing missing values
######
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# recheck if there is any missing value
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())
######
# Feature Engineering
######
# transform categorical data to str
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# label encoding: categorical label to index number
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
# add one feature, the total area of basement, first and second floor areas of each house
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# shape        
print('Shape all_data: {}'.format(all_data.shape))
### Skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))
#==============================================================================
 #plot the skewed varible 
#colRank = 1
#colname = skewness.head(50).index[colRank]
#sns.distplot(train[colname] , fit=norm)
## Get the fitted parameters used by the function
#(mu, sigma) = norm.fit(train[colname])
#print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
#plt.ylabel('Frequency')
#plt.title(colname + ' distribution')
#fig = plt.figure()
#res = stats.probplot(train[colname], plot=plt)
#plt.show()
#plt.close()
#==============================================================================
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    all_data[feat] += 1
#==============================================================================
# #check whether skewness is fixed
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# print("after fix skewness:")
# print(skewness.head(10))
# colRank = 0
# colname = skewness.head(50).index[colRank]
# sns.distplot(train[colname] , fit=norm)
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train[colname])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title(colname + ' distribution')
# fig = plt.figure()
# res = stats.probplot(train[colname], plot=plt)
# plt.show()
#==============================================================================
# dummy encoding: one-hot
all_data = pd.get_dummies(all_data)
print(all_data.shape)
# check the data for all numerical values
if(len(all_data[~all_data.applymap(np.isreal).all(1)])==0):
    print ("all data are numerical")
else:
    print ("following data are not numerical:")
    print (all_data[~all_data.applymap(np.isreal).all(1)])
#####################
# Modelling
#####################
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# get the train and test data
train = all_data[:ntrain]
test = all_data[ntrain:]
#Validation function
n_folds = 5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
######
# BASIC MODELS
######
## Notice that here the outliers are removed by RobustScaler() in the pipeline!
######
# Lasso Regression
######
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
######
# Elastic Net Regression :
######
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
######
# Kernel Ridge Regression :
######
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
######
# Gradient Boosting Regression :
######
## With huber loss that makes it robust to outliers
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
######
# Xgboost :
######
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)
######
# LightGBM
######
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
######
# Base models scores
######
#==============================================================================
# score = rmse_cv(lasso)
# print("\nLasso score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
# score = rmse_cv(ENet)
# print("ElasticNet score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
# score = rmse_cv(KRR)
# print("Kernel Ridge score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
# score = rmse_cv(GBoost)
# print("Gradient Boosting score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
# score = rmse_cv(model_xgb)
# print("Xgboost score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
# score = rmse_cv(model_lgb)
# print("LGBM score: {:.4f} (+/-{:.4f})\n" .format(score.mean(), score.std()))
#==============================================================================
######
# Stacking Models
######
######
# Simple: Averaged base models class
######
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the prediction
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

#averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
#score = rmse_cv(averaged_models)
#print(" Averaged base models score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
#averaged_models_all = AveragingModels(models = (ENet, GBoost, KRR, lasso,model_xgb,model_lgb))
#score = rmse_cv(averaged_models_all)
#print(" Averaged all base models score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))
######
# Less simple Stacking : Adding a Meta-model
######
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train, holdout in kfold.split(X, y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[train], y[train])
                y_pred = instance.predict(X[holdout])
                out_of_fold_predictions[holdout, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
score = rmse_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} (+/-{:.4f})".format(score.mean(), score.std()))
#####################
# Prediction
#####################
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
######
# StackedRegressor:
######
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmse(y_train, stacked_train_pred))
######
# XGBoost:
######
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmse(y_train, xgb_train_pred))
######
# LightGBM:
######
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmse(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''

print('RMSE score on train data:')
print(rmse(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.10 + lgb_train_pred*0.20 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.10 + lgb_pred*0.20
# make submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
print ("Finished...")