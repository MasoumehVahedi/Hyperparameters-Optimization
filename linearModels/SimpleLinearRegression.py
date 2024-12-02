import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def multivariateLinearRegression(df):
    X = df[["AMS", "ADNN", "VMS", "VDNN"]]
    Y = df[["T", "Bf"]]  # Multivariate target

    # Fit the model
    model = LinearRegression()
    model.fit(X, Y)

    # Make predictions
    predictions = model.predict(X)
    df["T_pred"], df["Bf_pred"] = predictions[:, 0], predictions[:, 1].astype(int)

    # Calculate R-squared for each target variable
    r2_T = r2_score(df["T"], df["T_pred"])
    r2_Bf = r2_score(df["Bf"], df["Bf_pred"])

    # Calculate RMSE for each target variable
    rmse_T = np.sqrt(mean_squared_error(df["T"], df["T_pred"]))
    rmse_Bf = np.sqrt(mean_squared_error(df["Bf"], df["Bf_pred"]))

    # Print accuracy metrics
    print("R-squared for T:", r2_T)
    print("R-squared for Bf:", r2_Bf)
    print("RMSE for T:", rmse_T)
    print("RMSE for Bf:", rmse_Bf)

    # Print model coefficients
    print("Coefficients for T and Bf:", model.coef_)
    print("Intercepts for T and Bf:", model.intercept_)



def linearRegressor(df, test_data):
    X = df[["AMS", "ADNN", "VMS", "VDNN"]]
    Y = df["T"]
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    X_test = test_data[["AMS", "ADNN", "VMS", "VDNN"]]
    X_test = sm.add_constant(X_test)

    model_T = sm.OLS(Y, X).fit()
    print(model_T.summary())
    predicted_T = model_T.predict(X_test)
    df["T_pred"] = predicted_T

    # For BF
    y = df["Bf"]  # Dependent variable for BF
    model_BF = sm.OLS(y, X).fit()
    print(model_BF.summary())
    predicted_Bf = model_BF.predict(X_test)
    df["Bf_pred"] = predicted_Bf.astype(int)

    df.to_csv(f"predicted_results.csv", index=False)

    sns.regplot(x="T", y="AMS", data=df, order=1)
    plt.show()



def predValuesWithLinearCoefficients(test_data):
    import numpy as np

    # Coefficients and intercepts we got from linear regression model (multivariateLinearRegression() function)
    coefficients = np.array([[4.41042854, -0.03387944, 1.41470208, 0.14916355],
                             [3.21726237, -0.02797192, -0.06387725, -0.00629145]])
    intercepts = np.array([22.63356404, 50.23895363])

    X_test = test_data[["AMS", "ADNN", "VMS", "VDNN"]]
    predictions = np.dot(X_test, coefficients.T) + intercepts
    # Extract predicted T and Bf
    test_data["T_pred"] = predictions[:, 0]
    test_data["Bf_pred"] = predictions[:, 1].astype(int)
    test_data.to_csv(f"predicted_results.csv", index=False)






if __name__ == "__main__":
    query_name = "1"
    df = pd.read_csv(f"all_median_{query_name}%.csv")
    print(df.corr())
    x_test = pd.read_csv("All_corrected_measurements.csv")
    #linearRegressor(df, x_test)
    #multivariateLinearRegression(df)
    predValuesWithLinearCoefficients(x_test)