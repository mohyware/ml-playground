from utils import *

# Task #5: Regularized polynomial regression
def RegularizedRegression(X,X_val,t):
    from sklearn.linear_model import Ridge
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # monomials
    degree = 2
    X_train_monomials = monomials_poly_features(X, degree)
    X_val_monomials = monomials_poly_features(X_val, degree)
    X_train_poly = scaler.fit_transform(X_train_monomials,1)
    X_val_poly = scaler.transform(X_val_monomials)

    # Define lambda (alpha) values to search
    lambda_values = {'model__alpha': alphas}  # Testing values from 10⁻⁴ to 10⁴

    # Perform Grid Search with Cross Validation (4-fold)
    kf = KFold(n_splits=4, random_state=RANDOM_STATE, shuffle=True)

    model = Ridge()

    pipeline = Pipeline(steps=[("scaler", MinMaxScaler()), ('model', model)])

    grid_search = GridSearchCV(pipeline, lambda_values, scoring='neg_mean_squared_error',cv=kf)

    grid_search.fit(X_train_poly, t)

    # Get the best lambda
    best_lambda = grid_search.best_params_['model__alpha']
    print("Best Lambda (Regularization Parameter):", best_lambda)

    rmses = np.sqrt(-grid_search.cv_results_['mean_test_score'])
    model = grid_search.best_estimator_

    for (alpha, rmse) in zip(alphas, rmses):
        t_pred = model.predict(X_val_poly)
        print(f'alpha = {alpha} - rmse = {rmse}', end='')
        if alpha == grid_search.best_params_['model__alpha']:
            print('\t\t**BEST PARAM**')
        else:
            print()

if __name__ == "__main__":
    RegularizedRegression(X, X_val, t)