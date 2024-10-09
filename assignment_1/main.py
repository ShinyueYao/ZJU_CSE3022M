from data_loader import load_mnist_data, sample_images
from canopy import Canopy
from kmeans import KMeans
from visualization import visualize_centers

def main():

    # Load the MNIST dataset
    train_data = load_mnist_data()
    # Sample images from the dataset, 250 per class
    points = sample_images(train_data, samples_per_class=250) 

    # Initialize Canopy algorithm
    canopy = Canopy(t1=22.5, t2=12)
    canopy.fit(points)
    
    # Get "Rough clusters" from canopy
    clusters = canopy.get_clusters()
    
    # Extract initial centers from clusters
    initial_centers = [c['center'] for c in clusters]  

    # Initialize K-Means algorithm
    kmeans = KMeans(max_iters=100)

    # Perform K-means clustering
    labels, final_centers = kmeans.fit(points, initial_centers)  

    visualize_centers(final_centers)  # Visualize the final cluster centers

if __name__ == "__main__":
    main()
