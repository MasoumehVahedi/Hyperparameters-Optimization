import json
import random
import numpy as np

from GenerateMBRs import GenerateMBR


class GenerateDataset:
    def __init__(self, df, x_min, y_min, x_max, y_max):
        self.genMBR = GenerateMBR()
        self.df = df
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get_measurements(self):
        ################## Constants based on real dataset ###############
        # Mean and standard deviation for size
        AMS = self.df["MMS"].iloc[0]
        VMS = self.df["VarianceMMS"].iloc[0]
        # Mean and standard deviation for Distance parameter
        ADNN = self.df["MDNN"].iloc[0]
        VDNN = self.df["VarianceMDNN"].iloc[0]
        return ADNN, VDNN, AMS, VMS


    def save_mms_mdnn_values(self, i, AMS, ADNN):
        with open(f"NewDatasets2/AMS_ADNN_values{i}.json", "w") as f:
            json.dump({"AMS": AMS, "ADNN": ADNN}, f)


    def generate_random(self, base_value):
        #lower_bound = base_value * 0.9  # 10% lower than the base value
        #upper_bound = base_value * 1.1  # 10% higher than the base value

        lower_bound = base_value * 1.2  # 20% higher than the base value
        upper_bound = base_value * 1.4  # 40% higher than the base value
        return random.uniform(lower_bound, upper_bound)


    def generate_sub_datasets(self, AMS_base, VMS_base, ADNN_base, VDNN_base, iteration, num_datasets):
        """ This function Generates a number of datasets where AMS and ADNN are the same across all datasets,
            but each dataset has different VMS and VDNN values. Like this:

                      (AMS-0, VMS-0, ADNN-0, VDNN-0)
                      (AMS-0, VMS-1, ADNN-0, VDNN-1)
                      (AMS-0, VMS-2, ADNN-0, VDNN-2)
                             .....
        """
        for i in range(1, num_datasets + 1):
            # Generate Random Variances for Size and Distance for each dataset
            VMS = self.generate_random(VMS_base)
            VDNN = self.generate_random(VDNN_base)
            # Calculate the standard deviations
            std_dev_size = np.sqrt(VMS)
            std_dev_distance = np.sqrt(VDNN)
            scaled_SD_size = std_dev_size * 200

            ###### Generate MBRs ######
            bounding_boxes = self.genMBR.generateMBRdataset(ADNN_base, std_dev_distance, AMS_base, scaled_SD_size, self.x_min, self.x_max, self.y_min, self.y_max)
            transformed_mbrs = self.genMBR.generate_mirrored_mbrs(bounding_boxes)
            file_path = f"NewDatasets2/Group{iteration}_MBRdataset{i}"
            np.save(file_path, transformed_mbrs)  # save new MBR dataset
        self.save_mms_mdnn_values(iteration, AMS_base, ADNN_base)  # save MMS and MDNN values of each dataset


    def generate_new_datasets(self, total_groups, num_datasets_per_group):
        """ This function is the final function to generate a bunch of different datasets with fixed (AMS and ADNN) and different
            (VMS and VDNN) like this:

                 (AMS-0, VMS-0, ADNN-0, VDNN-0)
                 (AMS-0, VMS-1, ADNN-0, VDNN-1)
                 (AMS-0, VMS-2, ADNN-0, VDNN-2)
                    .....

                 (AMS-1, VMS-0, ADNN-1, VDNN-0)
                 (AMS-1, VMS-1, ADNN-1, VDNN-1)
                 (AMS-1, VMS-2, ADNN-1, VDNN-2)
                    .....

                 (AMS-2, VMS-0, ADNN-2, VDNN-0)
                 (AMS-2, VMS-1, ADNN-2, VDNN-1)
                 (AMS-2, VMS-2, ADNN-2, VDNN-2)
        """
        ADNN_base, VDNN_base, AMS_base, VMS_base = self.get_measurements()

        for group in range(1, total_groups + 1):
            ########## Generate Random Values ##########
            AMS = self.generate_random(AMS_base)
            VMS = self.generate_random(VMS_base)
            ADNN = self.generate_random(ADNN_base)
            VDNN = self.generate_random(VDNN_base)
            self.generate_sub_datasets(AMS, VMS, ADNN, VDNN, group, num_datasets_per_group)


    def load_mms_mdnn_values(self, i):
        with open(f"NewDatasets2/AMS_ADNN_values{i}.json", "r") as f:
            values = json.load(f)
        return values["AMS"], values["ADNN"]

