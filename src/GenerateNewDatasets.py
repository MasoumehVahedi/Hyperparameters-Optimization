import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generatingMBRs import GenerateMBR



class GenerateDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.genMBR = GenerateMBR()


    def generate_random(self, base_value, lower_ratio=1.4, upper_ratio=1.8):
        """Generate a random value within a range based on a given ratio."""
        lower_bound = base_value * lower_ratio
        upper_bound = base_value * upper_ratio
        return np.random.uniform(lower_bound, upper_bound)


    def generate_datasets(self, AMS_values, ADNN_values, num_vms_variations, num_vdnn_variations, measurements_df, output_csv_path):
        """Generate datasets for each systematic combination of AMS and ADNN, with VMS and VDNN variations."""
        all_measurements = pd.DataFrame()
        x_min, x_max = -180, 180
        y_min, y_max = -90, 90

        # Create systematic combinations of AMS and ADNN (7 AMS x 7 ADNN = 49 combinations)
        for ams_idx, AMS in enumerate(AMS_values, start=1):
            for adnn_idx, ADNN in enumerate(ADNN_values, start=1):
                print(f"Generating datasets for combination AMS_{ams_idx}, ADNN_{adnn_idx}: AMS={AMS}, ADNN={ADNN}")

                # Get the corresponding bounding box limits from the original measurements DataFrame by row index
                row = measurements_df.iloc[ams_idx - 1]  # Get AMS row
                VMS_base = row["VarianceMMS"]
                VDNN_base = row["VarianceMDNN"]

                # Generate 25 datasets with VMS and VDNN variations for each AMS-ADNN pair
                for vms_index in range(1, num_vms_variations + 1):
                    for vdnn_index in range(1, num_vdnn_variations + 1):
                        # Generate random VMS and VDNN variations
                        VMS = self.generate_random(VMS_base)
                        VDNN = self.generate_random(VDNN_base)

                        # Generate standard deviations based on variances
                        std_dev_size = np.sqrt(VMS)
                        std_dev_distance = np.sqrt(VDNN)
                        scaled_SD_size = std_dev_size * 50  # Use smaller scaling factor

                        # Generate MBRs
                        bounding_boxes = self.genMBR.generateMBRdataset(AMS, scaled_SD_size,x_min, x_max)
                        transformed_mbrs = self.genMBR.generateMirroredMBRs(bounding_boxes)

                        # Validate MBRs after mirroring
                        valid_mbrs = [mbr for mbr in transformed_mbrs if self.genMBR.validateMBR(*mbr)]

                        # Create a descriptive name for the dataset including AMS, ADNN, VMS, VDNN
                        file_name = f"AMS{ams_idx}_ADNN{adnn_idx}_VMS{vms_index}_VDNN{vdnn_index}.npy"
                        file_path = os.path.join(self.base_dir, file_name)
                        #print(len(valid_mbrs))
                        # Save dataset (MBRs)
                        np.save(file_path, valid_mbrs)

                        # Save the generated values for this dataset
                        dataset_df = pd.DataFrame({
                            'Dataset': [file_name],  # Save the file name as the dataset identifier
                            'AMS': [AMS],
                            'ADNN': [ADNN],
                            'VMS': [VMS],
                            'VDNN': [VDNN]
                        })

                        # Add to the main DataFrame
                        all_measurements = pd.concat([all_measurements, dataset_df], ignore_index=True)

                        print(f"Saved dataset: {file_name}")

        # Save all measurements, including VMS and VDNN, to a CSV file
        all_measurements.to_csv(output_csv_path, index=False)
        print(f"All measurements with VMS and VDNN saved to {output_csv_path}")



