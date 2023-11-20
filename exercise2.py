import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sigmoid activation function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# one-layer neural network
def neuralNet(data, weights, bias):
    y = np.dot(data, weights) + bias
    output = sigmoid(y)
    
    classifications = []
    for i in range(len(output)):
        # if activation function is >= .5, round up
        if (output[i] >= .5): classifications.append(1)
        else: classifications.append(0)
    return classifications, output

def plotBound(weights, bias, color='violet', label=None):
    b = -(bias/weights[1])
    m = -(weights[0]/weights[1])

    plt.axline([0,b], slope=m, color=color, label=label)
    
def plotSurface(df, weights, bias):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data
    data = df[(df['species'] == 'versicolor') | (df['species'] == 'virginica')].iloc[:,2:4]
    length = np.array(data['petal_length'])
    width = np.array(data['petal_width'])
    X = np.linspace(np.amin(length), np.amax(length))
    Y = np.linspace(np.amin(width), np.amax(width))
    x, y = np.meshgrid(X, Y)
    z = np.array(neuralNet(np.c_[x.ravel(), y.ravel()], weights, bias)[1])
    
    # Plot the surface.
    surf = ax.plot_surface(x, y, z.reshape(x.shape), cmap='viridis')
    fig.colorbar(surf, ax=ax)
    ax.set_zlim(0, 1.01)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Neural Network Over Input Space')
    
    
def irisPlot(df):
    versicolor = df[(df['species'] == 'versicolor')]
    virginica = df[(df['species']  == 'virginica')]
    plt.scatter(versicolor['petal_length'], versicolor['petal_width'], color='blue', label='Versicolor')
    plt.scatter(virginica['petal_length'], virginica['petal_width'], color='red', label='Virginica')
    plt.legend(loc='upper left')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Versicolor and Virginica Data')

def main():
    df = pd.read_csv("./irisdata.csv")
    '''
    2a.
    '''
    irisPlot(df)
    # plt.show()
    
    '''
    2c.
    '''
    weights = np.array([.06, .1])
    bias = -.45    
    plt.title("Versicolor and Virginica Data w/ Decision Boundary")
    plotBound(weights, bias)
    plt.xlim([0,7])
    plt.ylim([0,2.6])
    plt.text(1, .5, "Weights: " + str(weights) + "\nBias: " + str(bias))
    plt.show()
    
    '''
    2d.
    '''
    # weights = np.array([10,12])
    # bias = -66
    # plotSurface(df, weights, bias)
    # plt.show()
    
    '''
    2e.
    '''
    # points=[[3.2,1.0], [7.0,2.3], [4.5,1.5], [4.5,1.7], [5.9, 1.5]]
    # classifications = neuralNet(points, weights, bias)[0]
    # for i in range(len(points)):
    #     color = 'blue'
    #     if classifications[i] == 1:
    #         color = 'red'
    #     plt.plot(points[i][0], points[i][1], marker='o', color=color)
    # plotBound(weights, bias)
    # plt.legend(['Versicolor', 'Virginica'])
    # plt.title('Classification Example')
    # plt.show()
    
if __name__ == "__main__":
    main()