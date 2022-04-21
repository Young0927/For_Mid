import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
from scipy import stats

'''Seaborn was used to draw a scatterplot. 
The data is a csv file in which Process_Time, WAC_Time, and throughput are recorded.
'''

df = pd.read_csv("RawData_sample.csv")
'''
 sns.scatterplot() can draw simple scatterplot.
The default treatment of the hue (and to a lesser extent, size) semantic, 
if present, depends on whether the variable is inferred to represent “numeric” or “categorical” data. 
In particular, numeric variables are represented with a sequential colormap by default, 
and the legend entries show regular “ticks” with values that may or may not exist in the data. 
This behavior can be controlled through various parameters, as like Process_Time or WAC_Time.

g = sns.scatterplot(x = "JIT1", y="Throughput",hue="Process_Time", data = df)
g.set(title = "Throughput against JIT1 with different Process Time")
plt.show()



g = sns.lmplot(x = "JIT1", y="Throughput", data = df.query("WAC_Time > 40"), ci =None, size=7)
g.set(title = "Throughput against JIT1 with different Process Time when WAC time above 40 second")
plt.show()

g = sns.relplot(x="JIT1", y="Throughput", hue="Process_Time",
    aspect=.75, linewidth=2.5,
    kind="line", data=df.query("WAC_Time > 40"))
g.set(title = "Throughput against JIT1 with different Process Time")
plt.show()


g = sns.lmplot(x = "JIT1", y="Throughput",order=2, data = df.query('WAC_Time == 40 and Process_Time == 120'), ci =None, size=7)
g.set(title = "Throughput against JIT1 with different Process Time when WAC time above 40 second")
plt.show()
'''


'''
Once drawn a plot using FacetGrid.map() (which can be called multiple times), 
there are also a number of methods on the FacetGrid object for manipulating the figure at a higher level of abstraction.
The most general is FacetGrid.set(), and there are other more specialized methods like FacetGrid.set_axis_labels().

sort = df.query("Process_Time > 50")
g = sns.FacetGrid(sort, col="Process_Time", col_wrap=3, height=2, ylim=(30, 80))
g.map(sns.pointplot, "JIT1", "Throughput",color="#334488", ci=None)
g.set(xticks=[20, 50,70], yticks=[30, 50, 80])
g.figure.subplots_adjust(wspace=.1, hspace=.3)
plt.show()
'''
def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

'''
The FacetGrid class is useful when you want to visualize the distribution of a variable or the relationship 
between multiple variables separately within subsets of your dataset. This time chart shows Throughput againest JIT1 
for each process time. 
'''


g = sns.FacetGrid(df, hue="WAC_Time", col="Process_Time", height=4, col_wrap=3)
g.map(qqplot, "JIT1", "Throughput")
g.set(xticks=[20, 50,70], yticks=[30, 50, 80])
g.add_legend()
plt.show()

'''
lmplot() is convenient interface to fit regression models across conditional subsets of a dataset.
polynomial regression for below chart
'''

sns.lmplot(x="JIT1", y="Throughput", data=df.query("Process_Time == 180"),order=2,hue="WAC_Time",ci=None, scatter_kws={"s": 80})
plt.show()