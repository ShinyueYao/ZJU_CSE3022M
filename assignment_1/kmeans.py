import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, max_iters=100):
        self.max_iters = max_iters

    def fit(self, points, centers):
        
        '''Loop for each iteration'''
        for iteration in range(self.max_iters):
            # Create cluster labels for each point
            labels = []  
            
            '''Assign each point to the nearest cluster center'''
            for point in points:
                # Calculate distances to all centers
                distances = [np.linalg.norm(point - center) for center in centers]  
                # Get the index of the closest center
                labels.append(np.argmin(distances))  

            new_centers = []  # List to hold the new cluster centers
            
            '''Update cluster centers '''
            for i in range(len(centers)):
                cluster_points = [points[j] for j in range(len(points)) if labels[j] == i]
                if cluster_points:
                    # Compute new center as the mean of points
                    new_centers.append(np.mean(cluster_points, axis=0))  
                else:
                    # Keep old center if no points are assigned
                    new_centers.append(centers[i])  

            ''' Visualize the points and centers using PCA'''
            self.visualize_pca(points, labels, new_centers, iteration)

            '''Check for convergence'''
            if np.array_equal(new_centers, centers):
                print(f"K-means converged after {iteration + 1} iterations.")
                break
            
            '''Update centers for the next iteration'''
            centers = new_centers  
            
            print(f"Iteration {iteration + 1}: Updated centers.")
        
        '''Return labels as integers and final centers'''
        return np.array(labels, dtype=int), centers

    def visualize_pca(self, points, labels, centers, iteration):
        
        '''Initialize PCA for 2D reduction, '''
        pca = PCA(n_components=2)
        reduced_points = pca.fit_transform(points)
        reduced_centers = pca.transform(centers) 

        plt.figure(figsize=(8, 6))
        
        # Plot points
        scatter = plt.scatter(reduced_points[:, 0], reduced_points[:, 1], c=labels, cmap='rainbow', alpha=0.5, s=10)
        # Plot centers  
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='black', marker='x', s=100, label='Centers')  
        
        plt.title(f'PCA Scatter Plot at Iteration {iteration + 1}')  
        plt.xlabel('PCA Component 1')  
        plt.ylabel('PCA Component 2')  
        plt.colorbar(scatter, label='Cluster Label')  
        plt.legend() 
        plt.savefig(f'./output/iteration/pca_scatter_plot_iteration_{iteration + 1}.png')  
        plt.show(block=False)  # Display plot
        plt.pause(1)
        plt.close()
