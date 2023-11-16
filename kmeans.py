import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

iris_data = pd.read_csv("./irisdata.csv")
maxIterations = 1000

# general k-means algorithm
def kmeans(data, k):
    data = data.select_dtypes(include=['int64', 'float64']).to_numpy()
    sample_numbers = np.random.choice(len(data), size=k)
    centroids = np.array(data[sample_numbers])
    centroids.sort(0)
    
    iterations = 0
    oldCentroids = None
    objVals = []
    
    while not shouldStop(oldCentroids, centroids, iterations):
        oldCentroids = np.array(centroids)
        iterations += 1

        labels, val = getLabels(data, centroids)
        objVals.append(val)
        
        centroids = updateCentroids(data, labels, k)
    return centroids, objVals

# stop algorithm if there are too many iterations, or if centroids do not change
def shouldStop(oldCentroids, centroids, iterations):
    if (iterations > maxIterations): return True
    return (oldCentroids == centroids).all()

# calculate distance between points and centroids and assign nearest centroid to each point
def getLabels(data, centroids):
    labels = []
    val = 0
    for i in range(len(data)):
        point = data[i]
        label = -1
        minDist = sys.maxsize
        for j in range(len(centroids)):
            centroid = centroids[j]
            dist = eucDist(point, centroid)
            if (dist < minDist):
                minDist = dist
                label = j
        labels.append(label)
        val += minDist
    return labels, val
  
def updateCentroids(data, labels, k):
    newCentroids = []
    for i in range(k):
        cluster = np.array([0.0] * len(data[0]))
        count = 0
        for j in range(len(labels)):
            if (labels[j] == i):
                cluster += data[j]
                count += 1
        if count != 0:
            cluster /= count
        else:
            cluster = data[random.randint(0, len(data))]
        newCentroids.append(cluster)
    return np.array(newCentroids)

# euclidean distance between data points 
def eucDist(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def main():
    k = 2
    clusters, objValues = kmeans(iris_data, k)
    plt.plot(range(1,len(objValues)+1), objValues, marker="o")
    plt.xticks(range(1,len(objValues)+1))
    plt.ylabel("Objective Function Value")
    plt.xlabel("Iteration")
    plt.title("Learning Curve for K = " + str(k))
    plt.show()
    
if __name__ == "__main__":
    main()