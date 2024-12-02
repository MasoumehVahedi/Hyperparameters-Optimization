import os
import random
import numpy as np
import pandas as pd
from CostFunction import OptimalHyperparameters
from SPLindex.SPLI import SPLI



def main(query_path, query_name):
    ########### Set up ############
    current_dir = os.getcwd()
    datasets_dir = os.path.join(current_dir, "Generated_Datasets")
    measurements_csv_path = os.path.join(current_dir, f"predicted_results_{query_name}.csv")

    # Define results directory path
    save_dir = os.path.join(current_dir, f"pred_results_{query_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ########### Load query ############
    query_ranges = np.load(query_path, allow_pickle=True)
    # Load measurements CSV to get dataset names and properties
    df = pd.read_csv(measurements_csv_path)

    # Add columns for results
    results = pd.DataFrame(columns=["Query", "AMS", "ADNN", "T", "Bf", "min_cost", "T_pred", "Bf_pred", "min_cost_pred"])

    # Iterate over each row of the DataFrame
    for idx, row in df.iterrows():
        # Extract AMS, ADNN, T, Bf, etc. for the current row
        ams_value = row['AMS']
        adnn_value = row['ADNN']
        t_value = row['T']
        bf_value = row['Bf']
        cost_value = row['Cost_history']

        # Get the dataset names from the current row
        row_datasets = row["Dataset"]
        # Split the concatenated dataset names by ', '
        dataset_names = row_datasets.split(", ")
        # Iterate over each dataset name in the list
        for dataset_name in dataset_names:
            # Construct the full file path for the dataset
            dataset_path = os.path.join(datasets_dir, dataset_name)

            if os.path.exists(dataset_path):
                polygons = np.load(dataset_path, allow_pickle=True)
                print(len(polygons))

                ########### Build SPLindex ###########
                print(f"-------- SPLindex building for {dataset_name} ---------")
                spli = SPLI(polygons, page_size=4096)

                ########### Range Query ###########
                print(f"-------- Range Query for {dataset_name} ---------")
                opt_hp_obj = OptimalHyperparameters()
                pred_T = row['Predicted_T']
                pred_Bf = int(row['Predicted_Bf'])  # Convert 'Predicted_Bf' to int
                pred_cost = opt_hp_obj.cost_function(polygons, pred_Bf, pred_T, spli, query_ranges)

                # Append the results for the current dataset to the DataFrame
                results = pd.concat(
                    [results, pd.DataFrame([{
                        "Query": query_name,
                        "AMS": ams_value,
                        "ADNN": adnn_value,
                        "T": t_value,
                        "Bf": bf_value,
                        "min_cost": cost_value,
                        "T_pred": pred_T,
                        "Bf_pred": pred_Bf,
                        "min_cost_pred": pred_cost
                    }])],
                    ignore_index=True
                )

            else:
                print(f"File not found: {dataset_path}")

    results.index = np.arange(1, len(results) + 1)
    # Save results with unique filename for each dataset-query combination
    results_file = os.path.join(save_dir, f"pred_cost_results_{query_name}.csv")
    results.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main(query_path, query_name)

