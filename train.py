import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import LogisticRegression;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.tree import DecisionTreeRegressor;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.neighbors import KNeighborsRegressor;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;
from lightgbm import LGBMRegressor

import tensorflow as tf
import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;
from tensorflow.keras.callbacks import EarlyStopping;

from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


import os;
from os import listdir;
from PIL import Image as PImage;
import cv2

ss = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip');
ss.head(20)



train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

train.head()



test.head(20)


features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip');
store = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')

features.head()


store.head()


train.shape, store.shape

((421570, 5), (45, 3))

train.isnull().sum()/train.shape[0]*100



test.isnull().sum()/test.shape[0]*100


newtrain = train.drop(columns = 'Date', axis =1);
newtest = test.drop(columns = 'Date', axis =1)

newtrain['IsHoliday'].unique()


h = {False : 1,  True : 2};
newtrain['IsHoliday'] = newtrain['IsHoliday'].map(h)
newtest['IsHoliday'] = newtest['IsHoliday'].map(h)

 

newtrain.head()



plt.figure(figsize = (16,9))
fineTech_appData3 = newtrain.drop(['Weekly_Sales'], axis = 1) # drop 'enrolled' feature
sns.barplot(fineTech_appData3.columns,fineTech_appData3.corrwith(newtrain['Weekly_Sales']))



newtrain = newtrain.drop(columns = 'IsHoliday', axis = 1);
newtest = newtest.drop(columns = 'IsHoliday', axis = 1);

x = newtrain.drop(columns = 'Weekly_Sales', axis =1);
y = newtrain['Weekly_Sales'];
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 55)

xx=  XGBRegressor();
xx.fit(xtrain, ytrain);
xx.score(xtest, ytest)

0.8831815747080758

xtrain


rfc = RandomForestRegressor();
rfc.fit(xtrain, ytrain);
rfc.score(xtest, ytest)



from lightgbm import LGBMRegressor
lgb = LGBMRegressor(n_estimators= 10000)
lgb.fit(xtrain,ytrain);
lgb.score(xtest, ytest)



knn = KNeighborsRegressor();
knn.fit(xtrain, ytrain);
knn.score(xtest, ytest)



dc = DecisionTreeRegressor();
dc.fit(xtrain, ytrain);
dc.score(xtest, ytest)



newtest.head()



model = rfc.predict(newtest);
modelData = pd.DataFrame(model, columns = ['Weekly_Sales']);
modelData.set_index('Weekly_Sales').to_csv('submission.csv')

