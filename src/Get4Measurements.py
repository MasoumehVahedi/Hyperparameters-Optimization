import numpy as np
import pandas as pd


class Measurements:
    def __init__(self, polygons):
        self.polygons = polygons
        self.data = self.computeMBB(self.polygons)

    def computeMBB(self, polygons):
        MBRs = []
        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            MBRs.append([minx, miny, maxx, maxy])
        return MBRs

    def calMinMBRsDistance(self, mbr1, mbr2):
        minx1, miny1, maxx1, maxy1 = mbr1
        minx2, miny2, maxx2, maxy2 = mbr2
        dist1 = np.sqrt((minx1 - minx2) ** 2 + (miny1 - miny2) ** 2)
        dist2 = np.sqrt((maxx1 - maxx2) ** 2 + (maxy1 - maxy2) ** 2)
        return min(dist1, dist2)


    def calMaxSizeMBR(self, mbr):
        minx, miny, maxx, maxy = mbr
        width = maxx - minx
        height = maxy - miny
        max_size = max(width, height)
        return max_size

    def calVariance(self, values, ddof=0):
        """ This function returns the variance of the Minimum Distances of MBRs
            ddof: to set the degrees of freedom that we want to use when calculating the variance.
            The choice of ddof (Delta Degrees of Freedom) depends on whether calculate a sample variance (ddof=1) or population variance (ddof=0)
        """
        n = len(values)
        mean = sum(values) / n
        return sum((x - mean) ** 2 for x in values) / (n - ddof)


    def saveMeasurements(self):
        min_distances = []
        max_sizes = []
        for i in range(len(self.data) - 1):
            current_mbr = self.data[i]
            next_mbr = self.data[i + 1]
            min_dist = self.calMinMBRsDistance(current_mbr, next_mbr)
            max_size = self.calMaxSizeMBR(self.data[i])
            min_distances.append(min_dist)
            max_sizes.append(max_size)

        avg_dist = sum(min_distances) / len(min_distances)
        avg_max_size = sum(max_sizes) / len(max_sizes)
        print(f"Mean Distances Nearest Neighbour = {avg_dist} meters")
        print(f"Mean Maximum Size = {avg_max_size} meters")

        # Calculate variance of minimum distances and maximum size
        variance_min_distances = self.calVariance(min_distances)
        variance_max_sizes = self.calVariance(max_sizes)
        print(f"Variance Distances Nearest Neighbour = {variance_min_distances} square meters (m²)")
        print(f"Variance Maximum Size = {variance_max_sizes} square meters (m²)")

        df = pd.DataFrame({
            'MDNN': [avg_dist],
            'MMS': [avg_max_size],
            'VarianceMDNN': [variance_min_distances],
            'VarianceMMS': [variance_max_sizes]
        })
        df.to_csv("mbr_measurements.csv", index=False)
        return df
