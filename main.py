import os
import sys
import random
import numpy as np
import pandas as pd
from src.CostFunction import OptimalHyperparameters
from SPLindex.SPLI import SPLI




def main(query_path, query_name):
    """
        Read all generated datasets and employ gradient descent to get the optimal results.
    """

    # set up
    INITIAL_T = 50
    INITIAL_B = 50

    current_dir = os.getcwd()
    datasets_dir = os.path.join(current_dir, "Generated_Datasets")
    measurements_csv_path = os.path.join(current_dir, "All_Datasets_Measurements_With_VMS_VDNN.csv")

    # Define results directory path
    save_dir = os.path.join(current_dir, f"results_{query_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ########### Load query ############
    query_ranges = np.load(query_path, allow_pickle=True)

    # Load measurements CSV to get dataset names and properties
    measurements_df = pd.read_csv(measurements_csv_path)

    # Add columns for results
    results = pd.DataFrame(columns=["Dataset", "Query", "AMS", "ADNN", "VMS", "VDNN", "T", "Bf", "Cost_history"])


    for dataset_name in sorted(os.listdir(datasets_dir)):
        if dataset_name.endswith(".npy"):
            # Load polygons (MBRs) from the dataset
            dataset_path = os.path.join(datasets_dir, dataset_name)
            if not os.path.exists(dataset_path):
                print(f"File not found: {dataset_path}!")
                continue
            polygons = np.load(dataset_path, allow_pickle=True)

            # Find the row corresponding to the dataset name in the measurements DataFrame
            row = measurements_df.loc[measurements_df['Dataset'] == dataset_name]

            # Extract AMS, ADNN, VMS, VDNN values
            AMS = row['AMS'].values[0]
            ADNN = row['ADNN'].values[0]
            VMS = row['VMS'].values[0]
            VDNN = row['VDNN'].values[0]

            ########### Build SPLindex ###########
            print(f"-------- SPLindex building for {dataset_name} ---------")
            spli = SPLI(polygons, page_size=4096)

            ########### Range Query ###########
            print(f"-------- Range Query for {dataset_name} ---------")
            initial_bf = INITIAL_B
            initial_T = INITIAL_T

            opt_hp_obj = OptimalHyperparameters()
            T, Bf, cost_history = opt_hp_obj.gradient_descent(
                polygons, initial_bf=initial_bf, initial_T=initial_T,
                spli=spli, range_queries=query_ranges, output_csv=f"all_iteration_results_{dataset_name}_{query_name}"
            )

            # Append the results for the current dataset to the DataFrame
            results = pd.concat(
                [results, pd.DataFrame([{
                    "Dataset": dataset_name,
                    "Query": query_name,
                    "AMS": AMS,
                    "ADNN": ADNN,
                    "VMS": VMS,
                    "VDNN": VDNN,
                    "T": T,
                    "Bf": Bf,
                    "Cost_history": cost_history
                }])],
                ignore_index=True
            )

    # Set the DataFrame index to start from 1 instead of 0
    results.index = np.arange(1, len(results) + 1)

    # Save results with unique filename for each dataset-query combination
    results_file = os.path.join(save_dir, f"results_{query_name}.csv")
    results.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <query_path> <query_name>")
        sys.exit(1)

    query_path = sys.argv[1]
    query_name = sys.argv[2]

    main(query_path, query_name)

