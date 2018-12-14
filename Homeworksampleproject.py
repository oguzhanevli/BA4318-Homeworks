import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm


df = pd.read_csv("Danish Kron.txt" , sep = '\t')
print(df.axes)

def zero(array, NEWVALUE):
    array[array == 0] = NEWVALUE

df_copy = df.copy()
df_asarray = np.asarray(df_copy.VALUE)
zero(df_asarray, np.nan)
df_copy['VALUE'] = df_asarray
df_copy['NEWVALUE'] = df_copy.VALUE.interpolate()
print(df_copy)

size = len(df_copy)
head = df_copy[0:5]
tail = df_copy[size-5:]
print("Head")
print(head)
print("Tail")
print(tail)

train = df_copy[0:size-201]
test = df_copy[size-200:]

df.DATE = pd.to_datetime(df.DATE, format = "%Y-%m-%d")
df_copy.index = df_copy.DATE
train.DATE= pd.to_datetime(train.DATE, format= "%Y-%m-%d")
train.index = train.DATE
test.DATE= pd.to_datetime(train.DATE, format = "%Y-%m-%d")
test.index = test.DATE

print("Naive")
dd=np.asarray(train.NEWVALUE)
y_hat=test.copy()
y_hat['naive']=dd[len(dd)-1]
rms=sqrt(mean_squared_error(test.NEWVALUE, y_hat.naive))
print("RMSE:" ,rms)

print("Simple NEWVALUE")
y_hat_avg= test.copy()
y_hat_avg['avg_forecast']=train['NEWVALUE'].mean()
rms= sqrt(mean_squared_error(test.NEWVALUE, y_hat_avg.avg_forecast))
print("RMSE ", rms)



print("Simple Exponantial Smoothing")
y_hat_avg = test.copy()
alpha = 0.2
fit2=SimpleExpSmoothing(np.asarray(train['NEWVALUE'])).fit(smoothing_level=alpha, optimized = False)
y_hat_avg['SES']= fit2.forecast(len(test))
rms=sqrt(mean_squared_error(test.NEWVALUE, y_hat_avg.SES))
print("RMSE: ", rms)



print("Holt")
sm.tsa.seasonal_decompose(train.NEWVALUE).plot()
result=sm.tsa.stattools.adfuller(train.NEWVALUE)

y_hat_avg=test.copy()
alpha=0.4
fit1=Holt(np.asarray(train['NEWVALUE'])).fit(smoothing_level= alpha, smoothing_slope =0.1)
y_hat_avg['Holt_linear']=fit1.forecast(len(test))
rms=sqrt(mean_squared_error(test.NEWVALUE, y_hat_avg.Holt_linear))
print("RMSE: ", rms)



print("Holt_Winters")
y_hat_avg=test.copy()
seasons=10
fit1=ExponentialSmoothing(np.asarray(train['NEWVALUE']), seasonal_periods=seasons ,trend='add' , seasonal='add' ).fit()
y_hat_avg['Holt_Winter']=fit1.forecast(len(test))
rms=sqrt(mean_squared_error(test.NEWVALUE, y_hat_avg.Holt_Winter))
print("RMSE: ", rms)
