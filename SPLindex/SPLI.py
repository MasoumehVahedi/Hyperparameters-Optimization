import gc
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from .utils import *



class SPLI:
    def __init__(self, polygons, page_size=4096):
        self.polygons = polygons
        self.page_size = page_size
        self.leaf_count = 0
        self.internal_count = 0


    def get_byte_size(self, polygon):
        return len(polygon) * 16

    def regression_model(self, X, page_numbers):
        y = page_numbers.reshape(-1, 1)
        regressor = LinearRegression().fit(X, y)
        y_pred = regressor.predict(X).reshape(-1)
        return np.maximum(y_pred, 0)

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj)

    @staticmethod
    def loads(serialized_data):
        return pickle.loads(serialized_data)

    def save_pages_to_disk(self, filename="pages.bin"):
        page_map = []
        with open(filename, "wb") as f:
            for page in self.pages:
                start_pos = f.tell()
                pickle.dump(page, f)
                end_pos = f.tell()
                page_map.append((start_pos, end_pos))
        with open("page_map.pkl", "wb") as f:
            pickle.dump(page_map, f)


    def get_disk_pages(self, sorted_clusters):
        byte_sizes_gen = (self.get_byte_size(polygon) for cluster in sorted_clusters for polygon in cluster)
        X = np.array([[byte_size] for byte_size in byte_sizes_gen])
        byte_locations = np.cumsum(X)
        page_numbers = byte_locations // self.page_size
        y_pred = self.regression_model(X, page_numbers)

        self.pages = []
        self.cluster_hash_tables = defaultdict(dict)

        i = 0
        for j, cluster in enumerate(sorted_clusters):
            polygon_page_nums_cluster = {}
            for k, bounding_box_polygon in enumerate(cluster):
                bbp_tuple = tuple(bounding_box_polygon)
                page_index = max(int(y_pred[i]), 0)
                while len(self.pages) <= page_index:
                    self.pages.append([])
                self.pages[page_index].append((i, self.dumps(bounding_box_polygon)))
                polygon_page_nums_cluster[bbp_tuple] = (bounding_box_polygon, page_index)
                i += 1
            self.cluster_hash_tables[j] = polygon_page_nums_cluster
        self.save_pages_to_disk()

        yield self.cluster_hash_tables

        del self.pages
        del self.cluster_hash_tables
        gc.collect()


    def search(self, node, z_range):
        result = set()
        if node is None:
            return None

        if node.z_ranges[0][0] <= z_range[0] and node.z_ranges[-1][1] >= z_range[1]:
            left_cluster_probs = self.search(node.left_child, z_range)
            result.add(left_cluster_probs)

        if node.z_ranges[0][0] > z_range[0] and node.z_ranges[-1][1] < z_range[1]:
            right_cluster_probs = self.search(node.right_child, z_range)
            result.add(right_cluster_probs)

        if node.z_ranges[0][0] > z_range[0] and node.z_ranges[-1][1] < z_range[1]:
            right_cluster_probs = self.search(node.right_child, z_range)
            result.add(right_cluster_probs)

        return result

    def pred_cluster_ids(self, node, z_range):
        if node.leaf_model is not None:
            self.leaf_count += 1
            for (z_min, z_max), cluster_id in zip(node.z_ranges, node.clusters):
                if z_min <= z_range[1] and z_max >= z_range[0]:
                    yield (cluster_id, 1 / len(node.labels))
        else:
            self.internal_count += 1
            if node.internal_model is not None and node.z_ranges[0][0] <= z_range[0] and node.z_ranges[-1][1] >= \
                    z_range[1]:
                X = np.array([[z_range[0], z_range[1]]])
                probs = node.internal_model.predict(X).flatten()
                left_cluster_probs = list(self.pred_cluster_ids(node.left_child, z_range))
                right_cluster_probs = list(self.pred_cluster_ids(node.right_child, z_range))

                for cluster_id in node.labels:
                    left_prob = sum(p for c_id, p in left_cluster_probs if c_id == cluster_id)
                    right_prob = sum(p for c_id, p in right_cluster_probs if c_id == cluster_id)
                    prob = probs[cluster_id] + left_prob + right_prob
                    yield (cluster_id, prob)
            else:
                if node.left_child and node.left_child.z_ranges[-1][1] >= z_range[0]:
                    yield from self.pred_cluster_ids(node.left_child, z_range)
                if node.right_child and node.right_child.z_ranges[0][0] <= z_range[1]:
                    yield from self.pred_cluster_ids(node.right_child, z_range)


    def pred_cluster_ids_for_point_query(self, node, point_query):
        if node.leaf_model is not None:
            # leaf node: return the first cluster ID where point_query falls within the z_range
            return [(cluster_id, 1 / len(node.labels)) for (z_min, z_max), cluster_id in zip(node.z_ranges, node.clusters)
                    if z_min <= point_query <= z_max]
        else:
            # internal node
            if node.z_ranges[0][0] <= point_query <= node.z_ranges[-1][1]:
                if node.internal_model is not None:
                    # the point_query falls within the z_range of the internal node
                    X = np.array([[point_query]])
                    probs = node.internal_model.predict(X).flatten()
                    cluster_probs = []
                    # search left child
                    if node.left_child and node.left_child.z_ranges[0][0] <= point_query <= \
                            node.left_child.z_ranges[-1][1]:
                        left_cluster_probs = self.pred_cluster_ids_for_point_query(node.left_child, point_query)
                        cluster_probs.extend(left_cluster_probs)
                    # search right child
                    if node.right_child and node.right_child.z_ranges[0][0] <= point_query <= \
                            node.right_child.z_ranges[-1][1]:
                        right_cluster_probs = self.pred_cluster_ids_for_point_query(node.right_child, point_query)
                        cluster_probs.extend(right_cluster_probs)
                    if cluster_probs:
                        return cluster_probs
                    # the point_query does not fall within the z_range of any child nodes
                    for cluster_id in node.labels:
                        prob = probs[cluster_id]
                        return (cluster_id, prob)

    def get_predict_clusters(self, model, z_range):
        return self.pred_cluster_ids(model, z_range)


    def get_range_query_result(self, query_rect, hash_tables):
        xim, xmax, ymin, ymax = query_rect
        query_result = []
        for cluster_polygons in hash_tables:
            for polygon_mbb, value in cluster_polygons.items():
                if polygon_mbb[2] > xim and polygon_mbb[0] < xmax and polygon_mbb[3] > ymin and polygon_mbb[1] < ymax:
                    query_result.append(value)
        return query_result


    def get_predict_point_clusters(self, model, z_range):
        return self.pred_cluster_ids_for_point_query(model, z_range)


    def get_point_query_result(self, query_point, hash_tables):
        # Given a rectangle with points (x1,y1) and (x2,y2) and assuming x1 < x2 and y1 < y2, a point (x,y) is within that rectangle if x1 < x < x2 and y1 < y < y2.
        for pred_clusters in hash_tables:
            for polygon_mbb, value in pred_clusters.items():
                if polygon_mbb[0] <= query_point[0] <= polygon_mbb[2] and polygon_mbb[1] <= query_point[1] <= polygon_mbb[3]:
                    yield value


