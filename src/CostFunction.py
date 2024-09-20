"""
   Here, our goal is to find OPTIMAL hyperparameters from the clustering algorithm.

   The clustering algorithm we use is "BIRCH" with two important Hyperparameters including:
   1- T: Threshold or diameter between two clusters
   2- Bf: Branching factor or the maximum number of objects in CF tree

   Hypothesis: We want to use Average Maximum Size (AMS) and Average Distance Nearest Neighborhood (ADNN)
               of an MBR dataset to determine optimal (T, Bf) that are INDEPENDENT from VARIATION of MS and DNN.
               In other words, we want to find the linearity function between each pair of (MS + DNN) and (T, Bf):

                                      <AMS, ADNN> ---> <T, Bf>

   Cost Function: We want to find the optimal (T, Bf) from clustering algorithm to have better index searching
                  from SPLindex model we created. SPLindex is a learned index structure for cluster of MBRs, so
                  it highly depends on the result of clustering algorithm to minimize search TIME. Therefore, we
                  aim to find the best hyperparameters of clustering algorithm to have the "Minimal Search Time"
                  for a query.

   Goal: Our goal is to find optimal (T, Bf) by Gradient Decsent of a range of Hyperparameters (T and Bfs).
         In this case, we want to iterate through different (T, Bf) to find the minimum Query Time which is
         our Cost Function.
"""

import time
from tqdm import tqdm

from sklearn.cluster import Birch
from collections import defaultdict

from SPLindex.utils import *
from SPLindex.ZAdress import MortonCode
from SPLindex.treeModel import TreeBuilder
from SPLindex.ConfigParam import Config



class OptimalHyperparameters:

    def get_clusters(self, data, bf, T):
        birch = Birch(branching_factor=bf, n_clusters=None, threshold=T).fit(data)
        cluster_labels = birch.labels_
        num_clusters = len(set(cluster_labels))
        print(f"Number of clusters = {num_clusters}")
        clusters = [[rectangle for rectangle in data[cluster_labels == n]] for n in range(num_clusters)]
        return clusters


    def sort_clusters_Zaddress(self, data, bf, T):
        self.clusters = self.get_clusters(data, bf, T)
        MBR_clusters = []
        for i, cluster in enumerate(self.clusters):
            mbb = calculate_bounding_box(cluster)
            if np.all(np.isfinite(mbb)):
                MBR_clusters.append(mbb)

        all_z_addresses = []
        for mbr in MBR_clusters:
            z_addresses = [(MortonCode().interleave_latlng(mbr[0][1], mbr[0][0])),
                           (MortonCode().interleave_latlng(mbr[1][1], mbr[1][0]))]
            all_z_addresses.append(z_addresses)

        z_ranges_sorted = sorted(all_z_addresses, key=lambda x: x[0])
        sorted_indices = [all_z_addresses.index(c) for c in z_ranges_sorted]
        sorted_clusters = [self.clusters[i] for i in sorted_indices]
        sorted_clusters_IDs = [i for i, _ in enumerate(sorted_clusters)]
        return z_ranges_sorted, sorted_clusters_IDs, sorted_clusters


    def measure_execution_query_time(self, spli, model, hash_tables, query_rect):
        ############## Measure time for searching tree model and intermediate filtering ##############
        start = time.time()
        z_min = MortonCode().interleave_latlng(query_rect[2], query_rect[0])
        z_max = MortonCode().interleave_latlng(query_rect[3], query_rect[1])
        z_range = [z_min, z_max]
        predicted_labels = spli.get_predict_clusters(model, z_range)
        hash_pred_clusters = []
        for label in predicted_labels:
            hash_pred_clusters.append(hash_tables.get(label[0]))
        query_results = spli.get_range_query_result(query_rect, hash_pred_clusters)
        #print(len(query_results))
        end = time.time()
        QueryTime = end - start
        ############## Get a constant refinement time for accelerating process ##############
        # 200 microseconds
        constant_refinement_time = 0.0002  # seconds
        ExeQueryTime = QueryTime + constant_refinement_time
        return ExeQueryTime


    def cost_function(self, data, bf, T, spli, range_queries):
        ################### Clustering ####################
        z_ranges_sorted, sorted_clusters_IDs, sorted_clusters = self.sort_clusters_Zaddress(data, bf, T)
        ################# Index Building ##################
        tree = TreeBuilder(global_percentage=0.05)  # Assuming a 5% error bound for illustration
        model = tree.buildTreeModel(z_ranges_sorted, sorted_clusters_IDs)
        hash_tables_generator = spli.get_disk_pages(sorted_clusters)
        hash_tables = defaultdict(dict)
        for new_hash_tables in hash_tables_generator:
            hash_tables.update(new_hash_tables)
        ################# Execute Query ###################
        total_exe_query_time = 0
        for query_rect in range_queries:
            query_time = self.measure_execution_query_time(spli, model, hash_tables, query_rect)
            total_exe_query_time += query_time
        return total_exe_query_time


    def approximate_gradient_T(self, data, bf, T, spli, range_queries, delta_T):
        """ Since the cost function (query execution time) is not analytically differentiable
            with respect to the clustering hyperparameter (e.g., threshold in BIRCH clustering),
            we can use "numerical differentiation" to approximate the derivative.
            Numerical differentiation approximates the derivative by evaluating the function at
            two closely spaced points and calculating the slope of the line connecting these points.

            delta_T: is the small perturbations used for numerical differentiation.
        """
        original_time = self.cost_function(data, bf, T, spli, range_queries)
        increased_time = self.cost_function(data, bf, T + delta_T, spli, range_queries)
        gradient_T = (increased_time - original_time) / delta_T    # slope
        print(f"Original Time: {original_time}, Increased Time: {increased_time}, Gradient T: {gradient_T}")
        return gradient_T


    def approximate_gradient_bf(self, data, bf, T, spli, range_queries, delta_bf):
        """ Since the cost function (query execution time) is not analytically differentiable
            with respect to the clustering hyperparameter (e.g., threshold in BIRCH clustering),
            we can use "numerical differentiation" to approximate the derivative.

            delta_bf: is the small perturbations used for numerical differentiation
        """
        original_time = self.cost_function(data, bf, T, spli, range_queries)
        increased_time = self.cost_function(data, bf + delta_bf, T, spli, range_queries)
        gradient_bf = (increased_time - original_time) / delta_bf    # slope
        print(f"Original Time: {original_time}, Increased Time: {increased_time}, Gradient T: {gradient_bf}")
        return gradient_bf


    def gradient_descent(self, data, initial_bf, initial_T, spli, range_queries, initial_learning_rate_T=5.5,
                         delta_T=20.10, delta_bf=5, iterations=5, tolerance=1e-6):
        """ This function takes in the values of the set,  as well the intial theta values(coefficients),
            the learning rate, and the number of iterations. The output will be a new set of coefficeients(T),
            optimized for making predictions, as well as the array of the cost as it depreciates on each iteration.
        """

        T = initial_T
        bf = initial_bf
        learning_rate_T = initial_learning_rate_T
        cost_history = []

        for i in tqdm(range(iterations)):
            grad_T = self.approximate_gradient_T(data, bf, T, spli, range_queries, delta_T)
            T -= learning_rate_T * grad_T  # Update rule for T
            T = max(1e-5, T)  # To make sure T remains within the valid range

            grad_bf = self.approximate_gradient_bf(data, bf, T, spli, range_queries, delta_bf)
            # Update rule for bf to manage discrete parameters like bf in situations where traditional gradient descent is less
            # effective due to the discrete nature of the parameter or when gradients are too small.
            if grad_bf < 0:
                bf_update = 5  # Increase bf if the gradient indicates a decrease in cost
            elif grad_bf > 0:
                bf_update = -5  # Decrease bf if the gradient indicates an increase in cost
            else:
                bf_update = 0  # No change if gradient is zero
            bf = max(10, bf + bf_update)

            cost = self.cost_function(data, bf, T, spli, range_queries)
            cost_history.append(cost)
            print(f"Iteration {i + 1}: T={T}, Bf={bf}, Cost={cost}")

            if i > 5 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
                print(f"Convergence reached based on cost change at iteration {i + 1}.")
                break

        return T, bf, cost_history












































































