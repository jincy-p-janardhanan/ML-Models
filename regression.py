from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, model_selection, metrics, ensemble, svm, neighbors

def estimator(data):

    (X, Y) = data
    (m, n) = X.shape
    cols = X.columns

    print(X.shape, type(X))
    print(Y.shape, type(Y))
    print(X.info())
    print(X.describe())
    print(X.isna().sum())

    def plot_data():
        for i in range(n):
            plt.subplot(1, 2, 1)
            plt.scatter(X[cols[i]], Y)
            plt.xlabel(cols[i])
            plt.ylabel('Price')

            plt.subplot(1, 2, 2)
            # plt.hist(X[cols[i]])
            plt.violinplot(X[cols[i]])
            plt.xlabel(cols[i])

            plt.show()

    plot_data() # plot original data

    # Log normalization
    logcols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    X[logcols] = X[logcols].apply(lambda x: np.log(x))

    # Add new feature
    X['Loc'] = X['Latitude'] * X['Longitude']
    
    # Mean normalization and feature scaling
    normcols = list(set(list(X.columns)) - set(logcols))
    X[normcols] = X[normcols].apply(lambda x:((x - x.mean())/x.std()))

    plot_data() # plot normalized data

    # Train, Validation and Test Split
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y,test_size=0.20, random_state=10) 
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X, Y,test_size=0.50, random_state=10)

    trained_models = []

    # Linear Regression
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, Y_train)

    trained_models.append(lr_model)

    def evaluate(model):
        Y_pred_train = model.predict(X_train)
        Y_pred_val = model.predict(X_val)

        train_errors = Y_train - Y_pred_train
        val_errors = Y_val - Y_pred_val

        train_mse = metrics.mean_squared_error(Y_train, Y_pred_train)
        val_mse = metrics.mean_squared_error(Y_val, Y_pred_val)

        print(f'{model.__class__}:')
        print(f'\tTraining MSE: {train_mse}')
        print(f'\tValidation MSE    : {val_mse}')

        plt.subplot(1, 2, 1)
        plt.hist(train_errors, color='blue')
        plt.title('Train errors')

        plt.subplot(1, 2, 2)
        plt.hist(val_errors, color='red')
        plt.title('val errors')

        plt.suptitle(f'{model.__class__}')
        plt.show()
        return val_mse

    scores = []
    scores.append(evaluate(lr_model))

    # Bagging Regression
    bag_model = ensemble.BaggingRegressor(random_state=10)
    bag_model.fit(X_train, Y_train)
    trained_models.append(bag_model)
    scores.append(evaluate(bag_model))

    # Random Forest Regression
    rf_model = ensemble.RandomForestRegressor(n_estimators=25, random_state=10)
    rf_model.fit(X_train, Y_train)
    trained_models.append(rf_model)
    scores.append(evaluate(rf_model))

    # SVM Regression
    svm_model = svm.LinearSVR(random_state=10)
    svm_model.fit(X_train, Y_train)
    trained_models.append(svm_model)
    scores.append(evaluate(svm_model))

    # KNN Regression
    knn_model = neighbors.KNeighborsRegressor()
    knn_model.fit(X_train, Y_train)
    trained_models.append(knn_model)
    scores.append(evaluate(knn_model))

    # Choose the best model (least mse score)
    best_model = trained_models[scores.index(min(scores))]
    Y_pred_test = best_model.predict(X_test)
    errors = Y_test - Y_pred_test
    test_mse = metrics.mean_squared_error(Y_test, Y_pred_test)
    plt.hist(errors)
    plt.xlabel('Test Error')
    plt.show()

    return best_model, test_mse


if __name__ == '__main__':
    data = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
    model, mse = estimator(data)
    print(f'Best model: {model.__class__}')
    print(f'Best test  MSE: {mse}')