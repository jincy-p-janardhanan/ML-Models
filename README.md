# ML-Models
For an exploratory study of various machine learning models. Analysis and precprocessing of data may often be avoided since this study focuses on *usage* of various sklearn models with default configurations.

## Regression
Explores performance of various regression models.<br>
After importing the `regression.py` module, you can call the function `estimator(X, Y)` and get the best regression model for the dataset `(X, Y)`.<br>
The parameter `X` is a pandas dataframe of input features and `Y` is a pandas series of the target variable.<br>
Running the `regression.py` module as it is will give the following sample output which uses the classic California Housing Prices Dataset.

### Sample Output

![image](https://user-images.githubusercontent.com/51118633/159784912-aebb2345-38f0-421d-ab52-b035f144b52c.png)
```
<class 'sklearn.linear_model._base.LinearRegression'>:
	Training MSE	: 0.4415847034707834
	Validation MSE	: 0.46836770999839755

```
![image](https://user-images.githubusercontent.com/51118633/159785276-360dc152-fe95-45c6-a197-0a016911ab1e.png)
```
<class 'sklearn.ensemble._bagging.BaggingRegressor'>:
	Training MSE	: 0.0495235228848702
	Validation MSE	: 0.2809632860237471 

```
![image](https://user-images.githubusercontent.com/51118633/159785399-735a9e9f-d5e2-4ae9-9878-b57158a10aa7.png)
```
<class 'sklearn.ensemble._forest.RandomForestRegressor'>:
	Training MSE	: 0.03442765202858801
	Validation MSE	: 0.25679743116303017 

```
![image](https://user-images.githubusercontent.com/51118633/159785523-449efd12-b391-4016-aead-38e3be120aaf.png)
```
<class 'sklearn.svm._classes.LinearSVR'>:
	Training MSE	: 2.126105308335331
	Validation MSE	: 2.0886867115616874  

```
![image](https://user-images.githubusercontent.com/51118633/159785609-416a68a0-724d-43e2-b53d-1339b6257c2d.png)
```
<class 'sklearn.neighbors._regression.KNeighborsRegressor'>:
	Training MSE	: 0.2680161881869106
	Validation MSE	: 0.42012969031690883

```
![image](https://user-images.githubusercontent.com/51118633/159785982-119b2a40-396c-4966-b28e-5fee87ef2ecf.png)
```
Best model: <class 'sklearn.ensemble._forest.RandomForestRegressor'>
Test  MSE: 0.25317961771776054
```
In the current setting, this output is reproducible when California Housing Prices Dataset and the chosen random seed (10) are used.4

## Classification
Explores performance of various classification models.<br>
After importing the `classification.py` module, you can call the function `estimator(X, Y)` and get the best classification model for the dataset `(X, Y)`.<br>
The parameter `X` is a pandas dataframe of input features and `Y` is a pandas series of the target label.<br>
(Multi-class classification is assumed.)<br>
Running the `classification.py` module as it is will give the following sample output which uses the classic 20-News-Groups Dataset.

### Sample Output

![image](https://user-images.githubusercontent.com/51118633/159798849-edf99b0c-4c65-423d-a056-02f570710dbc.png)
```
<class 'sklearn.linear_model._logistic.LogisticRegression'>:
	Training F1 score	: 0.892072528976155
	Validation F1 score	: 0.8261910103702513
```
![image](https://user-images.githubusercontent.com/51118633/159798952-6f70afdf-6192-48ad-b0b4-2a127c71eb75.png)
```
<class 'sklearn.ensemble._bagging.BaggingClassifier'>:
	Training F1 score	: 0.9883881204315708
	Validation F1 score	: 0.7191521215859012 
```
![image](https://user-images.githubusercontent.com/51118633/159799021-5cc425b3-8bf0-4efd-99ce-063f2d92e275.png)
```
<class 'sklearn.ensemble._forest.RandomForestClassifier'>:
	Training F1 score	: 0.999933668957722
	Validation F1 score	: 0.8263376379540432 
```
![image](https://user-images.githubusercontent.com/51118633/159799142-07f0bd66-debb-4126-acc6-a2440f934872.png)
```
<class 'sklearn.svm._classes.LinearSVC'>:
	Training F1 score	: 0.9911449973287118
	Validation F1 score	: 0.9085398890414774 
```
![image](https://user-images.githubusercontent.com/51118633/159799242-398d4dc1-9a6d-4ea6-bc8e-92d71761cebc.png)
```
<class 'sklearn.neighbors._classification.KNeighborsClassifier'>:
	Training F1 score	: 0.7839324365743485
	Validation F1 score	: 0.627218602829993 
```
![image](https://user-images.githubusercontent.com/51118633/159799323-3a1da6da-19b4-4091-a8a0-ac4052f9b739.png)
```
Best model: <class 'sklearn.svm._classes.LinearSVC'>
Test  F1-score: 0.9111325934890819
```
