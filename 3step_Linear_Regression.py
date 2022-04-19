#experiment data package
from re import X
import xdrlib
from cv2 import DFT_COMPLEX_INPUT
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from correlation_pearson.code import CorrelationPearson

 
#data pre-process package
import numpy as np
import pandas as pd

#machine learning model set & valution package 
import scipy as sp
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#data visualization package
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pandas.plotting import scatter_matrix

#load data
df = pd.read_csv("RawData_sample.csv")


df = pd.DataFrame(df)

# split dataset 

X = df[['Process_Time', 'WAC_Time','JIT1' ,'Square_PW', 'PW']]
y = df['Throughput']


#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#Machine Learning model 
mlr = linear_model.LinearRegression()
mlr.fit(X = pd.DataFrame(X), y = y)
Prediction = mlr.predict(X = pd.DataFrame(X))

print('a value =', mlr.intercept_)
print('b value = ' , mlr.coef_)


my_throughput = [[100,30, 80, 70, 130]]
my_throughput = mlr.predict(my_throughput)

print('Throughput prediction = ', my_throughput)


#plt.scatter(y_test, y_predict, alpha=0.4)
#plt.xlabel("Actual Throughput")
#plt.ylabel("Predicted Throughput")
#plt.title("Multi Linear Regression")

#check relationship
#plt.scatter(df[['WAC_Time']],df[['Throughput']],alpha=0.6)

#R2  = check accuracy
residuals = y - Prediction
print(residuals.describe())

SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared =', R_squared)


print('score =' , mlr.score( X = pd.DataFrame(X), y = y))
print('Mean_Squared_Erroir = ' , mean_squared_error(Prediction, y))
print('RMSE =', mean_squared_error(Prediction, y )**0.5)



#RSS = ((y - y_predict) ** 2).sum()
#TSS = ((y - y.mean()) ** 2).sum()
#R_squared = 1-(RSS/TSS)

#print('R_squared = ', R_squared)

#print(mlr.score(x_train, y_train))

# Pearson product-moment correlation coefficient
print(df.corr())
sns.heatmap(df.corr(), annot=True)

plt.show()


#from the heatmap JIT1 coefficent is the largest with Process_Time
pearson_coef, p_value = stats.pearsonr(df['JIT1'], df['Process_Time'])
print("Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


#quick view
print(df)

#return a tuple representing the dimentionality of the dataframe
print(df.shape)


#plot the data
df.plot(kind='scatter',x="JIT1", y="Throughput")
plt.show()

X_axis=pd.DataFrame(df['JIT1'])
print(X_axis)
Y_axis=pd.DataFrame(df['Throughput'])
print(X_axis)

#build linear regression model
lm = linear_model.LinearRegression()
model = lm.fit(X_axis, Y_axis)

print(model.predict(X_axis))
print(model.coef_)
print(model.intercept_)

#evaluate model
print(model.score(X_axis, Y_axis))


#predict new value of Y
X_axis_new = []
exit()