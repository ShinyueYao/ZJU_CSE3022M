import numpy as np

class Canopy:
    def __init__(self, t1, t2): # t1 must bigger than t2
        self.t1 = t1  # Distance threshold for forming clusters
        self.t2 = t2  # Distance threshold for rejecting points
        self.clusters = []  # List of clusters

    def fit(self, points):
        for point in points:
            added = False  # Check if the point has been added to a cluster
            
            '''Check if the point can be added to an existing cluster'''
            for cluster in self.clusters:
                # Check distance against threshold t1
                if np.linalg.norm(point - cluster['center']) < self.t1:  
                    # Add the point to the cluster
                    cluster['points'].append(point)  
                    # Change the flag
                    added = True
                    break
            
            '''If the point was not added, then create a new cluster'''
            if not added:
                # Create a new cluster
                self.clusters.append({'center': point, 'points': [point]})

        print(f"Formed {len(self.clusters)} initial clusters using Canopy.")

    def get_clusters(self):
        # Return only clusters that have points
        return [c for c in self.clusters if len(c['points']) > 0]  
