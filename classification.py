from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, model_selection, metrics, ensemble, svm, neighbors

def estimator(X, Y):

    seed = 10

    # Train, Validation and Test Split
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y,test_size=0.20, random_state=seed) 
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val, Y_val,test_size=0.50, random_state=seed)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    def evaluate(model):
        # Get predictions by the model
        Y_pred_train = model.predict(X_train)
        Y_pred_val = model.predict(X_val)

        # Compute errors
        train_errors = (Y_train == Y_pred_train).astype(int)
        val_errors = (Y_val == Y_pred_val).astype(int)

        # Compute F1 score
        train_f1 = metrics.f1_score(Y_train, Y_pred_train, average='weighted')
        val_f1 = metrics.f1_score(Y_val, Y_pred_val, average='weighted')

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
        print(f'\tTraining F1 score\t: {train_f1}')
        print(f'\tValidation F1 score\t: {val_f1}', '\n\n')

        return val_f1

    trained_models = []
    scores = []

    # Logistic Regression
    lr_model = linear_model.LogisticRegression(multi_class='multinomial',  random_state=seed)
    lr_model.fit(X_train, Y_train)

    trained_models.append(lr_model)
    scores.append(evaluate(lr_model))

    # Bagging Classifier
    bag_model = ensemble.BaggingClassifier(random_state=seed)
    bag_model.fit(X_train, Y_train)
    trained_models.append(bag_model)
    scores.append(evaluate(bag_model))

    # Random Forest Classifier
    rf_model = ensemble.RandomForestClassifier(random_state=seed)
    rf_model.fit(X_train, Y_train)
    trained_models.append(rf_model)
    scores.append(evaluate(rf_model))

    # SVM Classifier
    svm_model = svm.LinearSVC(random_state=seed)
    svm_model.fit(X_train, Y_train)
    trained_models.append(svm_model)
    scores.append(evaluate(svm_model))

    # KNN Regression
    knn_model = neighbors.KNeighborsClassifier()
    knn_model.fit(X_train, Y_train)
    trained_models.append(knn_model)
    scores.append(evaluate(knn_model))

    # Choose the best model (least mse score)
    best_model = trained_models[scores.index(max(scores))]
    Y_pred_test = best_model.predict(X_test)
    errors = (Y_test == Y_pred_test).astype(int)
    test_f1 = metrics.f1_score(Y_test, Y_pred_test, average='weighted')

    # Plot test error
    plt.subplot(1, 1, 1)
    plt.hist(errors, color='purple')
    plt.title('Test Error')
    plt.suptitle(f'{best_model.__class__}')
    plt.show()

    return best_model, test_f1

if __name__ == '__main__':

    X, Y = datasets.fetch_20newsgroups_vectorized(subset='all', return_X_y=True, as_frame=True)
    model, f1 = estimator(X, Y)
    print(f'Best model: {model.__class__}')
    print(f'Test F1-score: {f1}')