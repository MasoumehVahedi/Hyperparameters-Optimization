import sys
import numpy as np
from sklearn.datasets import make_blobs


print("Python Path:", sys.path)


class GenerateMBR:
    def __init__(self, num_clusters=20, num_boxes_per_cluster=1000, num_sparse_boxes=5000):
        self.num_clusters = num_clusters
        self.num_boxes_per_cluster = num_boxes_per_cluster
        self.num_sparse_boxes = num_sparse_boxes

    def mirrorMBR(self, mbr, mirror_x, mirror_y):
        # Mirror an MBR along x and/or y axes and ensure valid bounds.
        xmin, ymin, xmax, ymax = mbr

        # Mirror directly across the axes
        if mirror_x:
            xmin, xmax = -xmax, -xmin
        if mirror_y:
            ymin, ymax = -ymax, -ymin

        # Ensure valid MBR coordinates (xmin < xmax, ymin < ymax)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        return (xmin, ymin, xmax, ymax)


    def generateMirroredMBRs(self, mbrs):
        # Process and mirror MBRs, returning mirrored MBRs directly as coordinates
        transformed_mbrs = []
        for mbr in mbrs:
            # Original MBR
            transformed_mbrs.append(mbr)
            # Apply mirroring to MBR
            transformed_mbrs.append(self.mirrorMBR(mbr, False, True))
            transformed_mbrs.append(self.mirrorMBR(mbr, True, False))
            transformed_mbrs.append(self.mirrorMBR(mbr, True, True))
        return transformed_mbrs


    def validateMBR(self, xmin, ymin, xmax, ymax, x_range=(-180, 180), y_range=(-90, 90)):
        """Ensure that the generated MBR is valid with xmin < xmax and ymin < ymax and within bounds."""
        x_min_bound, x_max_bound = x_range
        y_min_bound, y_max_bound = y_range
        if (
            xmin < xmax and ymin < ymax and  # Ensure it's a valid box
            x_min_bound <= xmin <= x_max_bound and
            x_min_bound <= xmax <= x_max_bound and
            y_min_bound <= ymin <= y_max_bound and
            y_min_bound <= ymax <= y_max_bound
        ):
            return True
        return False

    def generateMBRsize(self, MMS, scaled_SD_size):
        """Generate width and height for the MBR with positive values based on normal distribution."""
        width = max(0.5, np.abs(np.random.normal(MMS, scaled_SD_size)))
        height = max(0.5, np.abs(np.random.normal(MMS, scaled_SD_size)))
        return width, height

    def generateDenseClusters(self, cluster_points, MMS, scaled_SD_size):
        """Generate dense clusters of MBRs around given points from make_blobs."""
        boxes = []
        cluster_SD = 0.5  # A small SD for tight clusters around the blob points

        for center in cluster_points:
            for _ in range(self.num_boxes_per_cluster):
                distance = np.random.normal(0, cluster_SD)  # Tighter clustering around center
                angle = np.random.uniform(0, 2 * np.pi)
                xmin = center[0] + distance * np.cos(angle)
                ymin = center[1] + distance * np.sin(angle)

                # Generate size using the normal distribution
                width, height = self.generateMBRsize(MMS, scaled_SD_size)
                xmax = xmin + width
                ymax = ymin + height

                # Validate the coordinates before adding
                if self.validateMBR(xmin, ymin, xmax, ymax):
                    boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def generateSparseMBRs(self, MMS, scaled_SD_size):
        """Generate sparse MBRs spread across the entire defined area."""
        boxes = []
        x_range = (-180, 180)
        y_range = (-90, 90)
        for _ in range(self.num_sparse_boxes):
            xmin = np.random.uniform(x_range[0], x_range[1])
            ymin = np.random.uniform(y_range[0], y_range[1])

            # Generate size using the normal distribution
            width, height = self.generateMBRsize(MMS, scaled_SD_size)
            xmax = xmin + width
            ymax = ymin + height

            # Validate the coordinates before adding
            if self.validateMBR(xmin, ymin, xmax, ymax):
                boxes.append((xmin, ymin, xmax, ymax))
        return boxes

    def generateMBRdataset(self, MMS, scaled_SD_size, x_min, x_max):
        """ This function Generates a number of datasets where AMS and ADNN are the same across all datasets,
            but each dataset has different VMS and VDNN values. Like this:

                              (AMS-0, VMS-0, ADNN-0, VDNN-0)
                              (AMS-0, VMS-1, ADNN-0, VDNN-1)
                              (AMS-0, VMS-2, ADNN-0, VDNN-2)
                                     .....
        """
        bounding_boxes = []

        # Step 1: Generate clustered points using make_blobs
        cluster_centers, _ = make_blobs(n_samples=self.num_clusters, centers=self.num_clusters,
                                        cluster_std=5.0, center_box=(x_min + 5, x_max - 5),
                                        random_state=42)

        # Step 2: Generate dense clusters around the points from make_blobs
        bounding_boxes.extend(self.generateDenseClusters(cluster_centers, MMS, scaled_SD_size))

        # Step 3: Generate sparse MBRs spread out over the entire area
        bounding_boxes.extend(self.generateSparseMBRs(MMS, scaled_SD_size))

        return np.array(bounding_boxes)
