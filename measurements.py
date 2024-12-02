import os
import numpy as np
import pandas as pd

from src.Get4Measurements import Measurements



def measureAndSaveVariants(poly_datasets_path, output_csv_path):
    """
        Measure variants (AMS, ADNN, VMS, VDNN) from all datasets and save the results in a CSV file.
    """
    all_measurements = pd.DataFrame()
    
    for idx, poly_path in enumerate(poly_datasets_path):
        print(f"Processing dataset {idx + 1}...")
        polygons = np.load(poly_path, allow_pickle=True)
        # Get AMS, VMS, ADNN, VDNN for the real polygon dataset
        measurements = Measurements(polygons)
        dataset_df = measurements.getAndSaveMeasurements()
        dataset_df["Dataset"] = f"Dataset_{idx + 1}"

        # Append the current dataset measurements to the main DataFrame
        all_measurements = pd.concat([all_measurements, dataset_df], ignore_index=True)

    all_measurements.to_csv(output_csv_path, index=False)
    print(f"All measurements saved to {output_csv_path}")




















