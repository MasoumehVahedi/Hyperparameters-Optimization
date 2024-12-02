import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.GenerateNewDatasets import GenerateDataset




def generateDatasets():
    """
        Generate datasets based on values from a CSV file.
    """

    current_dir = os.getcwd()
    datasets_dir = os.path.join(current_dir, "Generated_Datasets")
    measurements_csv_path = os.path.join(current_dir, "All_measurements.csv")
    output_csv_path = os.path.join(current_dir, "All_Datasets_Measurements_With_VMS_VDNN.csv")

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    # Load the measurements data from CSV
    measurements_df = pd.read_csv(measurements_csv_path)
    print(measurements_df.columns)

    # Extract AMS and ADNN values (assuming there are 7 unique values for each)
    AMS_values = measurements_df['MMS'].values
    ADNN_values = measurements_df['MDNN'].values

    # Number of variations (5 VMS x 5 VDNN = 25 variations per AMS-ADNN combination)
    num_vms_variations = 5
    num_vdnn_variations = 5

    # Generating datasets
    obj_gen_dataset = GenerateDataset(base_dir=datasets_dir)
    obj_gen_dataset.generate_datasets(AMS_values, ADNN_values, num_vms_variations, num_vdnn_variations, measurements_df, output_csv_path)

