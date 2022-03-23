# Regression
Exploring performance of various machine models on the classic California Housing Price Dataset.
After importing the module, you can call the function `estimator(data)` and get the best regression model for that data.
The parameter `data` should be a tuple `(X, Y)` where `X` is a pandas dataframe of input features and `Y` is a pandas series of the target variable.

## Sample Output

```
<class 'sklearn.linear_model._base.LinearRegression'>:
        Trainingprec: 0.4415847034707834
        valprec    : 0.4546705439391979
<class 'sklearn.ensemble._bagging.BaggingRegressor'>:
        Trainingprec: 0.04983683924991127
        valprec    : 0.04969016781115988
<class 'sklearn.ensemble._forest.RandomForestRegressor'>:
        Trainingprec: 0.03880235652016182
        valprec    : 0.03904502709355855
<class 'sklearn.svm._classes.LinearSVR'>:
        Trainingprec: 0.450212183429223
        valprec    : 0.4649338474505642
<class 'sklearn.neighbors._regression.KNeighborsRegressor'>:
        Trainingprec: 0.22296202230415282
        valprec    : 0.22955876081621396
Best model: <class 'sklearn.ensemble._forest.RandomForestRegressor'>
Best test  MSE: 0.12847789092456738
```
In the current setting, this output is reproducible when California Housing Prices Dataset and the chosen random seed (10) are used.
