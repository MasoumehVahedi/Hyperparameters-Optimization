import os

from measurements import measureAndSaveVariants
from generator import generateDatasets



def main():
    PolygonDataset = ["PolyLand", "PolyWater", "PolyUniform", "PolyCorr", "PolyZipf"]

    current_dir = os.getcwd()
    datasets_dir = os.path.join(current_dir, "data")
    output_csv_path = os.path.join(current_dir, "All_Measurements.csv")

    # We have 7 polygon datasets saved as .npy files
    polygon_datasets_paths = [os.path.join(datasets_dir, f"{i}.npy") for i in PolygonDataset]

    ############### Step 1: Process all datasets and save the measurements to a CSV ###############
    measureAndSaveVariants(polygon_datasets_paths, output_csv_path)

    ############### Step 2: Generate New Datasets ###############
    generateDatasets()



if __name__ == "__main__":
    main()