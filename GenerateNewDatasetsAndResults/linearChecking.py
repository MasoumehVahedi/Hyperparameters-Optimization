import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import ast


################## Step 1- Calculate average of T and Bf and compare it with all records #######################
"""path = "dataset9/results9/datasets_results.csv"
df = pd.read_csv(path)

# Convert string representations to actual lists
df['Cost_history'] = df['Cost_history'].apply(ast.literal_eval)
# Apply 'min' to each row in the 'costs' column to find the minimum cost
df['minimum_cost'] = df['Cost_history'].apply(min)
# Group by 'AMS' and 'ADNN', then find the index of the minimum 'minimum_cost' within each group
idx = df.groupby(['AMS', 'ADNN'])['minimum_cost'].idxmin()
# Select the rows with these indices
result_df = df.loc[idx]
final_df = result_df[['AMS', 'ADNN', 'T', 'Bf', 'minimum_cost']]
#final_df.to_csv("final_results.csv")

#print(final_df[:10])

ams_mean = final_df['AMS'].mean()
adnn_mean = final_df['ADNN'].mean()
threshold_mean = final_df['T'].mean()
bf_mean = int(final_df['Bf'].mean())
cost_mean = final_df['minimum_cost'].mean()

print(f"AMS - Mean: {ams_mean}")
print(f"ADNN - Mean: {adnn_mean}")
print(f"Optimal Diameter - Mean: {threshold_mean}")
print(f"Optimal Bf - Mean: {bf_mean}")
print(f"Optimal cost - Mean: {cost_mean}")

# Creating a new row with the means, using None or np.nan for columns that don't get a mean value
mean_row = pd.DataFrame([{'AMS': ams_mean, 'ADNN': adnn_mean, 'T': threshold_mean, 'Bf': bf_mean, 'minimum_cost': cost_mean}]).set_index([pd.Index(['mean'])])
# For columns not mentioned, you can either omit them or set them to np.nan or None

# Appending the new row to the DataFrame
final_df = pd.concat([final_df, mean_row])
final_df.to_csv("final_results/final_results9.csv")
print(final_df)"""


################## Step 2- Get all mean in one pandas dataframe #######################
"""all_means_df = pd.DataFrame()
for i in range(1, 10):
    df = pd.read_csv(f"final_results/final_results{i}.csv", index_col=0)
    df_mean = df.loc[["mean"]]
    all_means_df = pd.concat([all_means_df, df_mean])

all_means_df.reset_index(drop=True, inplace=True)
all_means_df.sort_values(by=["minimum_cost"])
all_means_df.to_csv("new_all_means_df.csv")
print(all_means_df)"""

################## Step 3- Get linear regression between variables #######################
df = pd.read_csv("new_all_means_df.csv", index_col=0)
#print(df)
X = df[['AMS', 'ADNN']]  # Independent variables
Y = df['T']  # Dependent variable
X = sm.add_constant(X)  # Adds a constant term to the predictor

model_T = sm.OLS(Y, X).fit()
print(model_T.summary())
predicted_T = model_T.predict(X)
df['Predicted_T'] = predicted_T

# For BF
y = df['Bf']  # Dependent variable for BF
model_BF = sm.OLS(y, X).fit()
print(model_BF.summary())
predicted_Bf = model_BF.predict(X)
df['Predicted_Bf'] = predicted_Bf.astype(int)
print(predicted_Bf)

df.to_csv('new_predicted_results.csv')
print(df)


#sns.regplot(x="T", y="AMS", data=df, order=1)
#plt.show()
