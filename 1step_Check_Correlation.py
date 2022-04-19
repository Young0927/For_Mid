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


attributes = ['Process_Time','WAC_Time', 'JIT1', 'Throughput','PW','Square_PW']
scatter_matrix(df[attributes],figsize=(12, 8))

plt.show()
exit()