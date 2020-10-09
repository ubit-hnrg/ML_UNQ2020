# data manipulation
import pandas as pd
import numpy as np
from operator import itemgetter


## learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

## preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


### model performance
from sklearn import metrics

#ploting modules
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

## Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV



def my_train_test_plot(gridsearch,grid,hyp,ax = None,ylim = [0,1],ylabel = 'score',round_xticks = 3):
    sns.set_style("whitegrid")

    if ax == None:
        fig, ax = plt.subplots(1)        

    keys = grid[0]#.str.strplit('__')
    basename = [*keys][0].split('__')[0]
    name = basename+'__'+ hyp
    values = grid[0][name]    
        

    results = gridsearch.cv_results_
    cc = ['mean_train_score','std_train_score','mean_test_score','std_test_score']
    performance = pd.DataFrame(itemgetter(*cc)(results),index = cc,columns = values).transpose()

    performance[name] = [str(v) for v in values]

    perf_train = performance.mean_train_score
    perf_test = performance.mean_test_score

    ax.plot(performance[name], performance.mean_train_score)
    ax.plot(performance[name], performance.mean_test_score)

    ylow_train =   perf_train - performance.std_train_score
    yup_train = perf_train + performance.std_train_score

    ax.fill_between(performance[name], ylow_train, yup_train, alpha=0.5, edgecolor='lightgray', facecolor='lightgray')

    ylow_test =   perf_test - performance.std_test_score
    yup_test = perf_test + performance.std_test_score

    ax.fill_between(performance[name], ylow_test, yup_test, alpha=0.5, edgecolor='lightgray', facecolor='lightgray')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(hyp)
    if round_xticks != -1:
      ax.set_xticklabels(np.round(values,round_xticks), rotation=45)
    else:
      ax.set_xticklabels(values, rotation=45)

    ax.set_ylim(ylim)


def univariate_exploring(pipe,X,Y, hyp, range,cv = 5, ylim = [0,1]):
  ## defino grilla
  gridname = 'clasificador__%s'%hyp
  param_grid = [
    {gridname:range}
  ]

  search = GridSearchCV(pipe, param_grid, 
                        cv=cv,return_train_score = True,
                      scoring = 'average_precision').fit(X, Y)

  plot = my_train_test_plot(gridsearch=search,grid=param_grid,hyp=hyp,ylim = ylim,ylabel = hyp)
  return(plot)