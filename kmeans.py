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
    centroidProgression = []
    centroids = np.array(data[sample_numbers])
    centroids.sort(0)
    
    iterations = 0
    oldCentroids = None
    objVals = []
    
    while not shouldStop(oldCentroids, centroids, iterations):
        centroidProgression.append(centroids)
        oldCentroids = np.array(centroids)
        iterations += 1

        labels, val = getLabels(data, centroids)
        objVals.append(val)
        
        centroids = updateCentroids(data, labels, k)
    return centroids, objVals, centroidProgression

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
  
# update centroids as mean of points in each class
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
    clusters, objValues, centroidProgress = kmeans(iris_data, k)
    
    '''
    1d.
    '''
    converged = []
    for i in range(len(centroidProgress[0])):
        converged.append(clusters[i][2:4])
        
    plt.scatter(iris_data["petal_length"], iris_data["petal_width"])
    cX = []
    cY = []
    for i in range(len(converged)):
        cX.append(converged[i][0])
        cY.append(converged[i][1])
    # calculate slope and midpoint of line between centroids 
    m = []
    midpt = []
    for i in range(len(converged)-1):
        x1 = converged[i][0]
        x2 = converged[i+1][0]
        y1 = converged[i][1]
        y2 = converged[i+1][1]
        m.append((y2-y1)/(x2-x1))
        midpt.append([(x1+x2)/2, (y1+y2)/2])
    m = np.array(m)
    m *= -1
    np.power(m, -1)
    print(m)
    print(midpt)
    plt.plot(cX, cY, color="fuchsia",  marker='*', ls='none', ms=20, label="Converged")
    for i in range(len(m)):
        plt.axline(midpt[i], slope=m[i], color="red")
    plt.legend(loc="upper left")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Decision Boundaries for K = " + str(k))
    
    plt.show()
    
    '''
    1c.
    '''
    # beginning = []
    # intermediate = []
    # converged = []
    # print(clusters)
    # for i in range(len(centroidProgress[0])):
    #     beginning.append(centroidProgress[0][i][2:4])
    #     intermediate.append(centroidProgress[int(len(centroidProgress)/2)][i][2:4])
    #     converged.append(clusters[i][2:4])
        
    # plt.scatter(iris_data["petal_length"], iris_data["petal_width"])
    # bX = []
    # bY = []
    # iX = []
    # iY = []
    # cX = []
    # cY = []
    # for i in range(len(beginning)):
    #     bX.append(beginning[i][0])
    #     bY.append(beginning[i][1])
    #     iX.append(intermediate[i][0])
    #     iY.append(intermediate[i][1])
    #     cX.append(converged[i][0])
    #     cY.append(converged[i][1])
    # plt.plot(bX, bY, color="black",  marker='*', ls='none', ms=20, label="Beginning")
    # plt.plot(iX, iY, color="lime",  marker='*', ls='none', ms=20, label="Intermediate")
    # plt.plot(cX, cY, color="fuchsia",  marker='*', ls='none', ms=20, label="Converged")
    # plt.legend(loc="upper left")
    # plt.xlabel("Petal Length")
    # plt.ylabel("Petal Width")
    # plt.title("Cluster Center Progression for K = " + str(k))
    
    # plt.show()
    
    '''
    1b.
    '''
    # plt.plot(range(1,len(objValues)+1), objValues, marker="o")
    # plt.xticks(range(1,len(objValues)+1))
    # plt.ylabel("Objective Function Value")
    # plt.xlabel("Iteration")
    # plt.title("Learning Curve for K = " + str(k))
    # plt.show()
    
    
    
if __name__ == "__main__":
    main()