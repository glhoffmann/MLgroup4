import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from numpy import loadtxt, savetxt


def color_signature(filename, label, colors = 3):
    try:
        image = plt.imread(filename)
        img = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=colors, n_init=10, random_state=0)
        kmeans.fit(img)
        l = [label]
        l = np.array(l)
        x = np.append(l, kmeans.cluster_centers_[[i for i in range(colors)]])
        return x
    except ValueError:
        # Move the erroneous file to the ./train/error directory
        error_dir = os.path.join(os.path.dirname(filename), "error")
        os.makedirs(error_dir, exist_ok=True)
        os.rename(filename, os.path.join(error_dir, os.path.basename(filename)))
        print(f"Moved {filename} to {error_dir}")


def create_color_signature_list(directory_list, colors=3):
    color_signature_list = []
    directory_labels = []
    for directory in directory_list:
        for filename in tqdm(os.listdir(directory), desc=f"Processing directory {directory}", unit="file"):
            if filename.endswith(".png"):
                filepath = os.path.join(directory, filename)
                path = os.path.dirname(directory)
                l = os.path.basename(path)
                color_signature_list.append(color_signature(filepath, l, colors))
    return color_signature_list


# Example usage
directories = ["./data/train/Apple_png/"]#, "./data/train/Banana_png/", "./data/train/cherry_png/", "./data/train/avocado_png/", "./data/train/kiwi_png/", "./data/train/mango_png/", "./data/train/orange_png/", "./data/train/pinenapple_png/", "./data/train/strawberries_png/", "./data/train/watermelon_png/"]
d = create_color_signature_list(directories)
data = np.array(d)
savetxt("data.csv", data, delimiter = ",")

#break
data = loadtxt('data.csv', delimiter=",")
data = np.array(color_signature_list)
#print(data)

labels = np.array([x[0] for x in d])
data = np.array([x[1:] for x in d])

#model = sklearn.cluster.k_means(data, 3)
#clusters = dict(zip(labels, model[1]))
kmeans = KMeans(n_clusters = 10, n_init = 10, random_state = 0).fit(data)
cluster_labels = []

for i in range(10):
    cluster_data = data[kmeans.labels_ == i]
    cluster_labels.append(labels[kmeans.labels_ == i])
    label_counts = {label: np.sum(cluster_labels[i] == label) for label in np.unique(cluster_labels[i])}
    print(f"Cluster {i}: {len(cluster_labels[i])} points")
    for label, count in label_counts.items():
        print(f"\t{count} points from label '{label}'")
