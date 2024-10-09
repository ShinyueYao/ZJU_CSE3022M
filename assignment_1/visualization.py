import matplotlib.pyplot as plt

def visualize_centers(centers):
    plt.figure(figsize=(12, 6))
    
    # Plot each cluster center as a grayscale image
    for i, center in enumerate(centers):
        plt.subplot(1, len(centers), i + 1)
        plt.imshow(center.reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.title('Cluster Centers')
    plt.savefig('./output/centroids/cluster_centers.png')
    plt.show()
    plt.pause(2)
    plt.close()
    print("Cluster centers saved as './output/centriods/cluster_centers.png'.") 
