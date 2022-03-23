from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, model_selection, metrics, ensemble, svm, neighbors

def estimator(X, Y):

    # Train, Validation and Test Split
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y,test_size=0.20, random_state=10) 
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val, Y_val,test_size=0.50, random_state=10)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    def evaluate(model):
        # Get predictions by the model
        Y_pred_train = model.predict(X_train)
        Y_pred_val = model.predict(X_val)

        # Compute errors
        train_errors = Y_train - Y_pred_train
        val_errors = Y_val - Y_pred_val

        # Compute MSE
        train_mse = metrics.mean_squared_error(Y_train, Y_pred_train)
        val_mse = metrics.mean_squared_error(Y_val, Y_pred_val)

        # Plot errors
        plt.subplot(1, 2, 1)
        plt.hist(train_errors, color='blue')
        plt.title('Train Error')
        plt.subplot(1, 2, 2)
        plt.hist(val_errors, color='red')
        plt.title('Validation Error')
        plt.suptitle(f'{model.__class__}')
        plt.show()

        # Print results
        print(f'{model.__class__}:')
        print(f'\tTraining MSE\t: {train_mse}')
        print(f'\tValidation MSE\t: {val_mse}', '\n\n')

        return val_mse

    trained_models = []
    scores = []

    # Linear Regression
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, Y_train)

    trained_models.append(lr_model)
    scores.append(evaluate(lr_model))

    # Bagging Regression
    bag_model = ensemble.BaggingRegressor(random_state=10)
    bag_model.fit(X_train, Y_train)
    trained_models.append(bag_model)
    scores.append(evaluate(bag_model))

    # Random Forest Regression
    rf_model = ensemble.RandomForestRegressor(random_state=10)
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

    # Plot test error
    plt.subplot(1, 1, 1)
    plt.hist(errors, color='purple')
    plt.title('Test Error')
    plt.suptitle(f'{best_model.__class__}')
    plt.show()

    return best_model, test_mse

if __name__ == '__main__':

    (X, Y) = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
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

    # plot original data
    plot_data()

    # Add new feature
    X['Loc'] = X['Latitude'] * X['Longitude']

    # Log normalization
    logcols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    X[logcols] = X[logcols].apply(lambda x: np.log(x))

    # Mean normalization and feature scaling
    normcols = list(set(list(cols)) - set(logcols))
    X[normcols] = X[normcols].apply(lambda x:((x - x.mean())/x.std()))

    # plot normalized data
    plot_data()

    model, mse = estimator(X, Y)
    print(f'Best model: {model.__class__}')
    print(f'Test  MSE: {mse}')