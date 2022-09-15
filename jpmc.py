# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.linear_model import LinearRegression

folder = "E:/Acads/9th sem/jpmfq/"
train_update=pd.read_csv(folder+'train_GBDT9493.csv')
test=pd.read_csv(folder+'test_GBDT9493.csv')
fare=train_update['Fare']
train=train_update.drop('Fare', axis = 1)
correlations = train_update.corr()['Fare'].sort_values()

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(train, fare, test_size=0.3, random_state=0)

def GradientBooster(param_grid, n_jobs):
  estimator = GradientBoostingRegressor()
  #cv = ShuffleSplit(X_train.shape[0], test_size=0.2)
  classifier = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs,  verbose = 5)
  classifier.fit(X_train, y_train)
  print("Best Estimator learned through GridSearch")
  print(classifier.best_estimator_)
  return classifier.best_estimator_

print(__doc__)
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB 
from sklearn.datasets import load_digits 
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
  plt.figure()
  plt.title(title)
  if ylim is not None: 
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1) 
  test_scores_mean = np.mean(test_scores, axis=1) 
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
  plt.legend(loc="best")
  return plt

param_grid={'n_estimators':[100], 
            'learning_rate': [0.1], 
            'max_depth':[6],
            'min_samples_leaf':[3]
            }
n_jobs=6
best_est=GradientBooster(param_grid, n_jobs)
print("Best Estimator Parameters" )
print("---------------------------" )
print("n_estimators: %d" %best_est.n_estimators)
print("max_depth: %d" %best_est.max_depth) 
print("Learning Rate: %.1f" %best_est.learning_rate)
print("min_samples_leaf: %d" %best_est.min_samples_leaf) 
print("max_features: %.1f" %best_est.max_features)
print("Train R-squared: %.3f" %best_est.score(X_train,y_train))


title = "Learning Curves (Gradient Boosted Regression Trees)" 
estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth, learning_rate=best_est.learning_rate, min_samples_leaf=best_est.min_samples_leaf, max_features=best_est.max_features) 
plot_learning_curve(estimator, title, X_train, y_train, n_jobs=n_jobs) 
plt.show()

estimator.fit(train, fare)
y_pred=estimator.predict(test)
estimator.score(X_test,y_test)
np.savetxt('submitFinal.csv',y_pred,header='y_pred') 
