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
x = df[['Process_Time', 'WAC_Time','JIT1' ,'Square_PW', 'PW']]
y = df['Throughput']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

attributes = ['Process_Time','WAC_Time', 'JIT1', 'Throughput','PW','Square_PW']
scatter_matrix(df[attributes],figsize=(12, 8))

plt.show()

#check relationship
#plt.scatter(df[['WAC_Time']],df[['Throughput']],alpha=0.6)


mlr = LinearRegression()
mlr.fit(x_train.values, y_train.values)


'''
Simple linear regression is described by the linear equation y = m*x + b that we can draw directly, 
so we called m the slope and b the intercept.
Once every variable x has its own m. So 'm' is regression coefficients.
'b' is constant.

below is to show regression coefficient
'''
print(mlr.coef_)

'''
In that case, let's draw a scatter plot in matplotlib to see the correlation between variables and Throughput value.
Looking at the two figures below, the length of the process time and the length of the WAC time seem to be irrelevant, 
but this is because it is a nonlinear relationship.
'''
plt.scatter(df[['Process_Time']],df[['Throughput']], alpha=0.4)
plt.title('Throughput against different ProcessTime',fontsize=15) ## 타이틀 설정
plt.xlabel("Process Time",fontsize=10)
plt.ylabel("Throughput",fontsize=10)
plt.show()

plt.scatter(df[['WAC_Time']],df[['Throughput']], alpha=0.4)
plt.title('Throughput against different WACTime',fontsize=15) ## 타이틀 설정
plt.xlabel("WAC Time",fontsize=10)
plt.ylabel("Throughput",fontsize=10)
plt.show()


Prediction = mlr.predict(x.values)
residuals = y - Prediction
print(residuals.describe())

'''RMSE values when expressing the accuracy of predictions for a specific value if accuracy cannot be correctly indicated to be judge.
In general, the lower the number, the higher the accuracy.
RMSE provides information about the size of the error, but not the size of the actual value of the data. 
Therefore, even if the RMSE values are the same, the performance of the regression equation should be evaluated differently if the actual data have different values.

MSE is used as a loss function for deep learning.
The loss function is a function that quantifies the difference between the actual value and the predicted value, 
and may reduce the value of the loss function by an optimization method such as the gradient descent method and optimize the performance of the model.

Score is the coefficient of determination means a measure of how well an estimated linear model fits a given data.

'''
SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST) ##The ratio of the size of the error to the variance of the data (change)
print('R_squared =', R_squared)


print('MSE(Mean_Squared_Error) = ' , mean_squared_error(Prediction, y))
print('RMSE(Root Mean Squared Error) =', mean_squared_error(Prediction, y)**0.5)
print('score =' , mlr.score(x_train, y_train))

# Pearson product-moment correlation coefficient
print(df.corr())
sns.heatmap(df.corr(), annot=True)

plt.show()


#from the heatmap JIT1 coefficent is the largest with Process_Time
pearson_coef, p_value = stats.pearsonr(df['JIT1'], df['Process_Time'])
print("Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


