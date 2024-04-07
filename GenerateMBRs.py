import math
import numpy as np



class GenerateMBR:
    def __init__(self, num_clusters=12, num_boxes_per_cluster=800, num_random_boxes=2000):
        self.num_clusters = num_clusters
        self.num_boxes_per_cluster = num_boxes_per_cluster
        self.num_random_boxes = num_random_boxes


    def mirror_mbr(self, mbr, mirror_x, mirror_y):
        """Mirror an MBR along x and/or y axes directly"""
        xmin, ymin, xmax, ymax = mbr

        # Mirror directly across the axes
        if mirror_x:
            xmin, xmax = -xmax, -xmin
        if mirror_y:
            ymin, ymax = -ymax, -ymin
        return (xmin, ymin, xmax, ymax)


    def generate_mirrored_mbrs(self, mbrs):
        """Process and mirror MBRs, returning mirrored MBRs directly as coordinates"""
        transformed_mbrs = []
        for mbr in mbrs:
            # Original MBR
            transformed_mbrs.append(mbr)
            # Apply mirroring to MBR
            transformed_mbrs.append(self.mirror_mbr(mbr, False, True))
            transformed_mbrs.append(self.mirror_mbr(mbr, True, False))
            transformed_mbrs.append(self.mirror_mbr(mbr, True, True))
        return transformed_mbrs


    def generateMBRsize(self, MMS, scaled_SD_size):
        """ Generate a minimum bounding box with different size based on the average of
            MBR in the real dataset (MMS).
        """
        width = np.random.normal(MMS, scaled_SD_size)
        height = np.random.normal(MMS, scaled_SD_size)
        return width, height

    def generateDenseClusters(self, center, num_boxes, SDNN, MMS, scaled_SD_size):
        """  Generate position with high density and normally distributed distances from the center
             Using a factor of the SDNN to scale the distribution down for clusters.

             Parameters:
                 - center: Centroid of a cluster we want to generate.
                 - num_boxes: The number of bounding boxes to generate.
                 - SDNN: Standard Deviation of Distances Nearest Neighbour.
                 - MMS: Mean Maximum Size.
                 - scaled_SD_size: Scale up the standard deviation to get larger size variations.
                 - cluster_std_factor: Smaller standard deviation for dense clusters.

             Returns:
                 - A list of minimum bounding boxes, each representing xmin, ymin, xmax, ymax.
        """
        boxes = []
        for _ in range(num_boxes):
            distance = np.random.normal(0, SDNN)
            angle = np.random.uniform(0, 2 * np.pi)  # 360 degree
            xmin = center[0] + distance * np.cos(angle)  # map from center
            ymin = center[1] + distance * np.sin(angle)

            # Generate size using the normal size distribution
            width, height = self.generateMBRsize(MMS, scaled_SD_size)
            xmax = xmin + width
            ymax = ymin + height

            boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def generateMBRdataset(self, MDNN, SDNN, MMS, scaled_SD_size, x_min, x_max, y_min, y_max):
        """ Using Average Maximum Size (MMS) and Average Distance Nearest Neighbourhood (MDNN) of a polygon dataset,
            and also Variance of MS + DNN, we generate a new dataset for MBRs.

            We want to have both dense and sparse position. So, using generateDenseClusters() we generate dense MBRs,
            and the rest will be sparse MBRs.

            Parameters:
                 - MDNN: Mean distance to nearest neighbor.
                 - SDNN: Standard Deviation of Distances Nearest Neighbour.
                 - MMS: Mean Maximum Size.
                 - scaled_SD_size: Scale up the standard deviation to get larger size variations.
                 - x_min, x_max, y_min, y_max: the largest and smallest value in the whole space for generating MBRs.

             Returns:
                 - An array of minimum bounding boxes from both dense and sparse MBRs, each representing xmin, ymin, xmax, ymax.
        """
        bounding_boxes = []
        ####################### Generate high density MBRs ########################
        # Generate random centroids for clusters
        cluster_centers = [(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)) for _ in range(self.num_clusters)]
        # print(cluster_centers)
        for center in cluster_centers:
            bounding_boxes.extend(self.generateDenseClusters(center, self.num_boxes_per_cluster, SDNN, MMS, scaled_SD_size))

        ######################## Generate sparse MBRs ########################
        random_box_distance = MDNN * 2  # Increase this to spread mbrs out more
        for _ in range(self.num_random_boxes):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(SDNN, random_box_distance)
            xmin = x_min + (x_max - x_min) * np.random.rand() + distance * np.cos(angle)
            ymin = y_min + (y_max - y_min) * np.random.rand() + distance * np.sin(angle)

            # Generate size using the normal size distribution
            width, height = self.generateMBRsize(MMS, scaled_SD_size)
            xmax = xmin + width
            ymax = ymin + height

            bounding_boxes.append((xmin, ymin, xmax, ymax))
        return np.array(bounding_boxes)
