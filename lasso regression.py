'''
author:Dani Atalla
Date:09/09/2021
time:12:29
 lasso Regression
'''

#Importing libraries.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

for i in range(2,16):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
print(data.head())

from sklearn.linear_model import Lasso


def lasso_regression(data, predictors, alpha, models_to_plot={}):
    # Fit the model
    lassoreg = Lasso (alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit (data[predictors], data['y'])
    y_pred = lassoreg.predict (data[predictors])

    # Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot (models_to_plot[alpha])
        plt.tight_layout ()
        plt.plot (data['x'], y_pred)
        plt.plot (data['x'], data['y'], '.')
        plt.title ('Plot for alpha: %.3g' % alpha)

    # Return the result in pre-defined format
    rss = sum ((y_pred - data['y']) ** 2)
    ret = [rss]
    ret.extend ([lassoreg.intercept_])
    ret.extend (lassoreg.coef_)
    return ret

#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)

coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
