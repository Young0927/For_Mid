import os
import sys
import joblib
import numpy as np
import pandas as pd
from joblib import dump
import subprocess as sp
from pprint import pprint
import matplotlib.pyplot as plt
from odc.io.cgroups import get_cpu_quota
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, f1_score
import multiprocessing
multiprocessing.cpu_count()
import re
import subprocess

import numpy as np
from sklearn import datasets
from sklearn.learning_curve import learning_curve
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn import preprocessing
import multiprocessing 
import matplotlib.pyplot as plt


    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, n_jobs= ncpus)
    
    
    plt.plot(train_sizes, train_scores.mean(axis=1), lebel = "Train score")
    plt.plot(train_sizes, test_scores.mean(axis=1), '--', lebel = "Test score")
    