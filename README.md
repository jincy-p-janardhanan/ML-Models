# Regression
Exploring performance of various machine models on the classic California Housing Price Dataset.
After importing the module, you can call the function `estimator(data)` and get the best regression model for that data.
The parameter `data` should be a tuple `(X, Y)` where `X` is a pandas dataframe of input features and `Y` is a pandas series of the target variable.

## Sample Output

![image](https://user-images.githubusercontent.com/51118633/159776518-491bd947-7a03-469f-990e-14ebbb39fac0.png)
```
<class 'sklearn.linear_model._base.LinearRegression'>:
	Training MSE	: 0.4415847034707834
	Validation MSE	: 0.4546705439391979 

```
![image](https://user-images.githubusercontent.com/51118633/159776627-621e5bb2-6820-459e-8e42-5fece70ba793.png)
```
<class 'sklearn.ensemble._bagging.BaggingRegressor'>:
	Training MSE	: 0.0495235228848702
	Validation MSE	: 0.049521583060808325 

```
![image](https://user-images.githubusercontent.com/51118633/159776723-0f85c725-b815-472c-b8b7-ceb60428b2ee.png)
```
<class 'sklearn.ensemble._forest.RandomForestRegressor'>:
	Training MSE	: 0.03866445255277847
	Validation MSE	: 0.03886070485938917 

```
![image](https://user-images.githubusercontent.com/51118633/159776869-7429b9eb-3b6e-4ea4-886e-a73fc263cdef.png)
```
<class 'sklearn.svm._classes.LinearSVR'>:
	Training MSE	: 2.126105308335331
	Validation MSE	: 2.127781037646729 

```
![image](https://user-images.githubusercontent.com/51118633/159776954-480e1e22-e3f2-4a14-96ef-b775fba900cb.png)
```
<class 'sklearn.neighbors._regression.KNeighborsRegressor'>:
	Training MSE	: 0.2680161881869106
	Validation MSE	: 0.2737135373102252 

```
![image](https://user-images.githubusercontent.com/51118633/159777052-49894f17-5e16-4b84-9f16-7c59c9d1cfec.png)
```
Best model: <class 'sklearn.ensemble._forest.RandomForestRegressor'>
Best test  MSE: 0.12870989410482483
```
In the current setting, this output is reproducible when California Housing Prices Dataset and the chosen random seed (10) are used.
