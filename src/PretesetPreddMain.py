import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GenerateNewDatasets import GenerateDataset
from CostFunction import OptimalHyperparameters
from SPLindex.SPLI import SPLI



def main():
    ########### Load data ############
    num_datasets = 10
    #num_datasets_per_group = 10
    query_path = "landuse_query_ranges_1%.npy"
    query_ranges = np.load(query_path, allow_pickle=True)
    df = pd.read_csv("RetesteResults/Prev_predicted_results.csv")

    Current_dir = os.getcwd()
    datasets_dir = os.path.join(Current_dir, "RetesteResults")
    save_dir = os.path.join(Current_dir, "RetesteResults/final_results")
    if not os.path.exists("RetesteResults/final_results"):
        os.makedirs(save_dir)

    results = pd.DataFrame(columns=["pred_T", "pred_Bf", "pred_cost"])


    for dataset in range(1, num_datasets):
        mbr_path = os.path.join(datasets_dir, f"PrevNewDatasets{dataset}/PrevNewDatasets{dataset}/Group{1}_MBRdataset{1}.npy")
        if not os.path.exists(mbr_path):
            print(f"File not found: {mbr_path}!")
            continue
        polygons = np.load(mbr_path, allow_pickle=True)
        print(len(polygons))
        # Load MMS and MDNN values for the current dataset
        #AMS, ADNN = df["AMS"], df["ADNN"]

        ########### Build SPLindex ###########
        print("-------- SPLindex building---------")
        spli = SPLI(polygons, page_size=4096)

        ########### Range Query ###########
        print("-------- Range Query ---------")
        opt_hp_obj = OptimalHyperparameters()
        if dataset <= len(df):
            row = df.iloc[dataset - 1]  # DataFrame indices are 0-based
            pred_T = row['Predicted_T']
            pred_Bf = int(row['Predicted_Bf'])  # Convert 'Predicted_Bf' to int
            pred_cost = opt_hp_obj.cost_function(polygons, pred_Bf, pred_T, spli, query_ranges)

        # Append the results for the current dataset to the DataFrame
        results = pd.concat([results, pd.DataFrame([{"pred_T": pred_T, "pred_Bf": pred_Bf, "pred_cost": pred_cost}])], ignore_index=True)

    # Set the DataFrame index to start from 1 instead of 0
    results.index = np.arange(1, len(results) + 1)
    results.to_csv(os.path.join(save_dir, "Prev_all_pred_results.csv"), index_label="Dataset")


if __name__ == "__main__":
    main()
