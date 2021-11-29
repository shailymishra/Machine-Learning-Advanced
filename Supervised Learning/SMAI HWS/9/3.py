

import numpy as np
import matplotlib.pyplot as plt


def getDataSamples(totalsamples,mu=0,sigma=1):
    return np.random.normal(mu, sigma, totalsamples)

def varyNoOfSet(total, startSet = 1, endSet = 300):
    datasamples = getDataSamples(total)
    noOfSets = np.arange(startSet,endSet+1)
    means = []
    variences = []
    for setNo in (noOfSets):
        sampleperset = total // setNo
        setsample = np.random.choice(datasamples,sampleperset )
        means.append(np.mean(setsample))
        variences.append(np.var(setsample))
    plt.plot(noOfSets,variences, label= "Constant k = 10000",linestyle='solid')
    plt.xlabel('s (No of Sets)', fontsize=18)
    plt.ylabel('Variance', fontsize=16)
    plt.title('Constant DataSample k = 10000, varying total no of sets')
    axes = plt.gca()
    axes.set_ylim([-1,4])
    plt.show()


def varyNoOfDatapoints(total_of_sets, startSet = 50, endSet = 3000):
    noOfDatapoints = np.arange(startSet,endSet+1)
    means = []
    variances = []
    for nodatapoints in (noOfDatapoints):
        datasamples = getDataSamples(nodatapoints)
        sampleperset = nodatapoints // total_of_sets
        setsample = np.random.choice(datasamples,sampleperset )
        means.append(np.mean(setsample))
        variances.append(np.var(setsample))
    plt.plot(noOfDatapoints,variances, label= "Constant s = 20",linestyle='solid')
    plt.xlabel('k (no of samples)', fontsize=18)
    plt.ylabel('Variance', fontsize=16)
    plt.title('Constant no of sets s = 20, varying Total no of data samples k')
    axes = plt.gca()
    axes.set_ylim([-1,4])
    plt.show()



def createSetOfSamples(no_of_sets,totalSamples):
    sampleperset = totalSamples // no_of_sets
    datasamples = getDataSamples(totalSamples)
    print('Total Taking ', no_of_sets , 'sets  of ' , totalSamples , 'samples of normal variable')
    for i in range(no_of_sets):
        setsample = np.random.choice(datasamples, sampleperset)
        print("mean of ", i , " is " , np.mean(setsample))


createSetOfSamples(20,1000)
varyNoOfDatapoints(20)
varyNoOfSet(10000)




