from utils import *

# Task #6: Lasso Selection
def LassoSelection():
    from sklearn.linear_model import Lasso , Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel

    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 

    # Split the data into training and validation sets
    X_train, X_val = X[:100, :], X[-100:, :]
    t_train, t_val = t[:100], t[-100:]

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Apply Lasso for Feature Selection (No Cross-Validation)
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_train_scaled, t_train)

    # Select features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    print("Selected Features:", selected_features)
    print(len(selected_features))
    print('+++++++++++++++++++++++++++++++++++++++++++++\n')

    alpha_indices_dct = {}

    for alpha in alphas:
        model = Lasso(fit_intercept = True, alpha = alpha, max_iter = 10000)
        selector = SelectFromModel(estimator=model)
        selector.fit(X_train_scaled, t_train)
        #print(selector.threshold_)
        flags =selector.get_support()
        indices = np.flatnonzero(flags)
        alpha_indices_dct[alpha] = indices

        pred_t = selector.estimator_.predict(X_val_scaled)
        lass_val_err = mean_squared_error(t_val, pred_t)

        print(f'alpha={alpha}, selects {len(indices)} features and has {lass_val_err} val error')
    print('+++++++++++++++++++++++++++++++++++++++++++++\n')

if __name__ == "__main__":
    LassoSelection()