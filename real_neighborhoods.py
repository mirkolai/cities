import geopandas as gpd
from shapely.geometry import MultiPoint
import networkx as nx
from infomap import Infomap

"""
This script applies the Infomap algorithm to a road network graph (G),
identifying meaningful clusters (neighborhoods) within it. 
The first four levels of hierarchy are saved.
"""

class RealNeighborhood:

    @staticmethod
    def compute_real_neighborhood(G, max_depth=3, speed=4.8, random_seed=87):
        """
        Computes the "real" neighborhood structure of a road network graph.

        Parameters:
        - G: networkx.Graph -> The road network graph
        - max_depth: int -> Maximum depth for Infomap clustering (default: 3)
        - speed: float -> Speed (in km/h) used to compute travel time (default: 4.8 km/h)
        - random_seed: int -> Seed for reproducibility (default: 87)

        Returns:
        - hulls: dict -> Convex and concave hulls of identified clusters
        - node_modules: dict -> Mapping of nodes to their respective clusters
        """
        print("Processing road network with", G.number_of_nodes(), "nodes...")

        # Mapping between real node IDs and Infomap-compatible indices
        mapping_realid2fakeid = {node: i for i, node in enumerate(G.nodes())}
        mapping_fakeid2realid = {i: node for node, i in mapping_realid2fakeid.items()}

        # Minimum percentage of nodes required for a component to be considered significant
        MIN_PERCENTAGE = 1
        total_nodes = G.number_of_nodes()

        # Assign travel time as edge weights (assuming constant speed)
        for u, v, d in G.edges(data=True):
            d['time_travel'] = float(d['length']) / (speed / 3.6)  # Convert km/h to m/s

        # Identify connected components in the graph
        connected_components = list(nx.connected_components(G.to_undirected()))

        # Filter out small components
        significant_components = [
            component for component in connected_components
            if len(component) / total_nodes * 100 >= MIN_PERCENTAGE
        ]

        print(f"Total connected components: {len(connected_components)}")
        print(f"Significant components (> {MIN_PERCENTAGE}% of nodes): {len(significant_components)}")

        hulls = {}  # Stores convex and concave hulls of clusters
        node_modules = {}  # Stores cluster assignments of nodes
        count_modules = {}  # Keeps track of cluster count per depth level

        # Iterate through each significant connected component
        for component_idx, component in enumerate(significant_components):
            print(f'Analyzing component {component_idx + 1}/{len(significant_components)}...')

            # Extract subgraph for the current component
            subgraph = G.subgraph(component)

            # Relabel nodes for compatibility with Infomap
            H = nx.relabel_nodes(subgraph, mapping_realid2fakeid, copy=True)

            # Initialize and run Infomap clustering
            im = Infomap(silent=True, seed=random_seed)
            im.add_networkx_graph(H, weight="time_travel")
            im.run()

            # Ensure max_depth does not exceed Infomap's max depth
            if max_depth > im.max_depth:
                max_depth = im.max_depth

            depth = 1
            while depth < max_depth:
                print("Depth level:", depth)
                if depth not in count_modules:
                    count_modules[depth] = 0

                # Retrieve node assignments at the current depth level
                nodes = im.get_nodes(depth_level=depth)
                num_clusters = len(set(im.get_modules(depth_level=depth).values()))
                print(f"Component {component_idx}, Depth {depth}, Clusters found: {num_clusters}")

                # Process clustering if enough distinct clusters are detected
                if num_clusters > 3:
                    modules = {}  # Stores clusters and their nodes

                    if depth not in node_modules:
                        node_modules[depth] = {}

                    for node in nodes:
                        real_id = mapping_fakeid2realid[node.node_id]
                        cluster_id = node.module_id + count_modules[depth]
                        node_modules[depth][str(real_id)] = cluster_id

                        if cluster_id not in modules:
                            modules[cluster_id] = []
                        modules[cluster_id].append((G.nodes[real_id]['x'], G.nodes[real_id]['y']))

                    # Compute convex and concave hulls for each cluster
                    if depth not in hulls:
                        hulls[depth] = {}

                    for module_id, points in modules.items():
                        if module_id not in hulls[depth]:
                            hulls[depth][module_id] = {}

                        geo_series = gpd.GeoSeries([MultiPoint(points)], crs='EPSG:4326')
                        hulls[depth][module_id]['convex_hull'] = geo_series.convex_hull
                        hulls[depth][module_id]['concave_hull'] = geo_series.concave_hull(ratio=0.1)

                    count_modules[depth] += len(modules)
                    depth = max_depth  # Stop processing further depths if valid clusters are found

                depth += 1

        return hulls, node_modules