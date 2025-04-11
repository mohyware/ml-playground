from utils import *

# Task #1 Normal Regression
def NormalRegression(X_train_scaled,X_val_scaled,t_train):
    result = train(X_train_scaled,t_train)
    eval(result,X_train_scaled,X_val_scaled)


# Task #2 Polynomial Feature Expansion with degree 3
def PolynomialRegression(X_train,X_val,t_train,degree = 3):
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.fit_transform(X_val)

    # Scale data
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_val_poly = scaler.transform(X_val_poly)

    result = train(X_train_poly,t_train)
    eval(result,X_train_poly,X_val_poly)

# Task #3 Polynomial Feature Expansion with monomials (No cross features)
def MonomialRegression(X_train,X_val,t_train,degree = 3):
    X_train_monomials = monomials_poly_features(X_train, degree)
    X_val_monomials = monomials_poly_features(X_val, degree)

    # Scale data
    X_train_poly = scaler.fit_transform(X_train_monomials)
    X_val_poly = scaler.transform(X_val_monomials)

    result = train(X_train_poly,t_train)
    eval(result,X_train_poly,X_val_poly)

# Task #4 Individual features Expansion with cross features
def CrossFeatures(X_train,X_val,t_train,degrees = [1,2,3]):
    for degree in degrees:
        print("Degree: ", degree)
        features = [0,3,6]
        for feature in features:
            print("Feature: ", feature)
            X_train_single = X_train[:, [feature]]
            X_val_single = X_val[:, [feature]]
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            X_train_poly = poly.fit_transform(X_train_single)
            X_val_poly = poly.fit_transform(X_val_single)

            # Scale data
            X_train_poly = scaler.fit_transform(X_train_poly)
            X_val_poly = scaler.transform(X_val_poly)

            result = train(X_train_poly,t_train)
            eval(result,X_train_poly,X_val_poly)
            print("\n")

if __name__ == "__main__":
    print("="*50)
    print("Task #1 Normal Regression")
    print("="*50)
    NormalRegression(X_train_scaled,X_val_scaled,t_train)

    print("\n" + "="*50)
    print("Task #2 Polynomial Feature Expansion with degree 3    Note: Polynomial then scale")
    print("="*50)
    PolynomialRegression(X_train,X_val,t_train)

    print("\n" + "="*50)
    print("Task #3 Polynomial Feature Expansion with monomials (No cross features)")
    print("="*50)
    MonomialRegression(X_train_scaled,X_val_scaled,t_train)

    print("\n" + "="*50)
    print("Task #4 Individual features Expansion with cross features")
    print("="*50)
    CrossFeatures(X_train,X_val,t_train)