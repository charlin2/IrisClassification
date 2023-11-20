import exercise2 as e2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

maxIterations = 1000

def meanSquaredError(data, weights, bias, classes):
    classifications = np.array(e2.neuralNet(data, weights, bias)[0])
    return np.sum((classifications - classes)**2)/(2*len(data))

def gradientStep(data, weights, bias, classes, epsilon):
    y = e2.sigmoid(np.dot(data, weights) + bias)
    step = []
    for i in range(len(weights)):
        step.append(np.sum((y-classes)*(y*(1-y))*data.iloc[:, i])/len(data))
    step = np.array(step)*epsilon
    return weights - step, bias - np.sum(y-classes)*epsilon

def gradientDescent(data, weights, bias, classes, epsilon):
    iterations = 0
    errorList = [meanSquaredError(data, weights, bias, classes)]
    weightList = [weights]
    biasList = [bias]
    while iterations < maxIterations and meanSquaredError(data, weights, bias, classes) > .01:
        iterations += 1
        weights, bias = gradientStep(data, weights, bias, classes, epsilon)
        errorList.append(meanSquaredError(data, weights, bias, classes))
        weightList.append(weights)
        biasList.append(bias)
    return np.array(errorList), np.array(weightList), np.array(biasList)

def getIrisClasses(df):
    classes = []
    for i in range(len(df)):
        if df.iloc[i]['species'] == 'versicolor':
            classes.append(0)
        else:
            classes.append(1)
    return np.array(classes[50:150])

def main():
    df = pd.read_csv("./irisdata.csv")
    classes = getIrisClasses(df)
    data = df[(df['species'] == 'versicolor') | (df['species'] == 'virginica')].iloc[:,2:4]
    
    '''
    3b.
    '''
    weights = np.array([.06, .1])
    bias = -.45  
    
    # weights = np.array([-1, 2])
    # bias = 1
    
    # e2.irisPlot(df)
    # e2.plotBound(weights, bias)
    # plt.xlim(left=2)
    # plt.ylim(bottom=0)
    # plt.title("Iris Data w/ Decision Boundary, MSE = " + str(meanSquaredError(data, weights, bias, classes)))
    # plt.text(5, .5, "Weights: " + str(weights) + "\nBias: " + str(bias))
    # plt.show()
    
    '''
    3e.
    '''
    # epsilon = .1 # no need for normalization since we are using MSE
    # weights = np.array([.05,.15])
    # bias = -.42
    # print(meanSquaredError(data, weights, bias, classes))
    # e2.irisPlot(df)
    # e2.plotBound(weights, bias, label='First Boundary')
    # newWeights, newBias = gradientStep(data, weights, bias, classes, epsilon)
    # print(meanSquaredError(data, newWeights, newBias, classes))
    # e2.plotBound(newWeights, newBias, 'yellow', 'Second Boundary')
    # plt.legend(loc='upper left')
    # plt.title('Decision Boundary After One Step of Gradient Descent')
    # plt.show()
    
    '''
    4b.
    '''
    # weights = np.array([.05,.15])
    # bias = -.42
    # errorList, weightList, biasList = gradientDescent(data, weights, bias, classes, epsilon=.01)
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(range(0, len(errorList)), errorList)
    # ax1.set_xlim(right=500)
    # ax1.set_title('MSE over Iterations')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('MSE')
    # e2.irisPlot(df)
    # e2.plotBound(weightList[len(weightList)-1], biasList[len(biasList)-1],
    #              label='Final Boundary', color='green')
    # e2.plotBound(weightList[500], biasList[500],
    #              label='Intermediate Boundary', color='pink')
    # e2.plotBound(weightList[0], biasList[0], label='First Boundary', color='red')
    # plt.legend()
    # plt.xlim(left=2)
    # plt.ylim(top=3)
    # plt.title("Decision Boundary Progress")
    # plt.show()
    
    '''
    4c.
    '''
    weights = np.array([np.random.uniform(-.2, .2),np.random.uniform(-.2, .2)])
    bias = np.random.uniform(-.1, .1)
    errorList, weightList, biasList = gradientDescent(data, weights, bias, classes, epsilon=.01)
    e2.irisPlot(df)
    e2.plotBound(weightList[len(weightList)-1], biasList[len(biasList)-1],
                 label='Final Boundary', color='green')
    e2.plotBound(weightList[500], biasList[500],
                 label='Intermediate Boundary', color='pink')
    e2.plotBound(weightList[0], biasList[0], label='First Boundary', color='red')
    plt.legend()
    plt.xlim(left=2)
    plt.ylim(top=3)
    plt.title("Random Weights, Initial Values:\n " + "Weights: " + str(weights) + "\nBias: " + str(bias))
    plt.show()
    
    
    


if __name__ == "__main__":
    main()
    