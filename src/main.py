import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GenerateNewDatasets import GenerateDataset
from CostFunction import OptimalHyperparameters
from SPLindex.SPLI import SPLI



def main():
    ########### Generate New Datasets ############
    num_datasets = 5
    num_datasets_per_group = 5
    df = pd.read_csv("mbr_measurements.csv")
    """ Define WGS84 (EPSG 4326) spatial bounds from the original polygon dataset:
            Calculate total bounding box:
            Longitude (X-coordinates): from 8.0764593 to 15.1917301
            Latitude (Y-coordinates): from 54.55947879929967 to 57.7487778006454
    """
    #x_min, y_min, x_max, y_max = 8.0764593, 54.55947879929967, 15.1917301, 57.7487778006454
    # Define Web Mercator spatial bounds from the original polygon dataset
    x_min, x_max = -180, 180
    y_min, y_max = -90, 90
    obj_gen_dataset = GenerateDataset(df, x_min, y_min, x_max, y_max)
    obj_gen_dataset.generate_new_datasets(total_groups=num_datasets, num_datasets_per_group=num_datasets_per_group)

    ########### Load data ############
    #query_path = "query_ranges_1%.npy"
    query_path = "landuse_query_ranges_1%.npy"
    query_ranges = np.load(query_path, allow_pickle=True)

    Current_dir = os.getcwd()
    datasets_dir = os.path.join(Current_dir, "NewDatasets2")
    save_dir = os.path.join(Current_dir, "results2")
    if not os.path.exists("results2"):
        os.makedirs(save_dir)

    results = pd.DataFrame(columns=["AMS", "ADNN", "T", "Bf", "Cost_history"])

    for group in range(1, num_datasets + 1):
        for dataset in range(1, num_datasets_per_group + 1):
            mbr_path = os.path.join(datasets_dir, f"Group{group}_MBRdataset{dataset}.npy")
            if not os.path.exists(mbr_path):
                print(f"File not found: {mbr_path}!")
                continue
            polygons = np.load(mbr_path, allow_pickle=True)
            print(len(polygons))
            # Load MMS and MDNN values for the current dataset
            AMS, ADNN = obj_gen_dataset.load_mms_mdnn_values(group)

            ########### Build SPLindex ###########
            print("-------- SPLindex building---------")
            spli = SPLI(polygons, page_size=4096)

            ########### Range Query ###########
            print("-------- Range Query ---------")
            # Pick random values for branding_factor (bf) and threshold to start with
            initial_bf = random.randint(50, 60)  # integer value
            #initial_T = random.uniform(0, 2)  # floating-point value
            initial_T = random.uniform(40, 50)
            # Apply gradient descent to the cost function (in our case is execution search time query)
            opt_hp_obj = OptimalHyperparameters()
            T, Bf, cost_history = opt_hp_obj.gradient_descent(polygons, initial_bf=initial_bf, initial_T=initial_T, spli=spli, range_queries=query_ranges)
            # print(f"Execution Time = {cost_history}")
            # print(f"Optimal Threshold = {T}")
            # print(f"Optimal Branching factor = {Bf}")

            # Append the results for the current dataset to the DataFrame
            results = pd.concat([results, pd.DataFrame([{"AMS": AMS, "ADNN": ADNN, "T": T, "Bf": Bf, "Cost_history": cost_history}])], ignore_index=True)

            # Plot the cost function
            plt.title('Cost Function')
            plt.xlabel('No. of iterations')
            plt.ylabel('Cost')
            plt.plot(cost_history)
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.savefig(os.path.join(save_dir, f"Cost_histogram_Group{group}_Dataset{dataset}.png"))

    # Set the DataFrame index to start from 1 instead of 0
    results.index = np.arange(1, len(results) + 1)
    results.to_csv(os.path.join(save_dir, "datasets_results.csv"), index_label="Dataset")


if __name__ == "__main__":
    main()
