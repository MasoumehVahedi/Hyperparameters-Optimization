import pandas as pd

df1 = pd.read_csv("new_all_pred_results.csv", index_col=0)
pred_cost = df1["pred_cost"][0:9].reset_index(drop=True)
print(df1)
pred_T = df1["pred_T"][0:9].reset_index(drop=True)
pred_Bf = df1["pred_Bf"][0:9].reset_index(drop=True)

df2 = pd.read_csv("new_all_means_df.csv", index_col=0)
df2["pred_T"] = pred_T
df2["pred_Bf"] = pred_Bf
df2["pred_minimum_cost"] = pred_cost

df2.to_csv("final_results/final_pred_result.csv")