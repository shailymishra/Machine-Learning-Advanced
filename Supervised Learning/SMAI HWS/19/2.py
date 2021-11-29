from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)

def MSEerr(y, pred):
    err = (1 / len(y)) * np.sum((y - pred) ** 2)
    return err


def hinge_loss( y,pred):
    err = (1 / len(y)) * np.sum(np.maximum(0, (1 - (y * pred))))
    print(err)
    return err

def MSErrDeri(y,pred):
    # print('loss shape', (y-pred).shape)
    return 2*(y-pred)

def hingeLossDeri(y,pred,x):
    loss = 0
    for index in range(len(y)):
        if(y[index]*pred[index] < 1):
            loss += (-pred[index]*x[index])
    # print('loss shape', loss.shape)
    return loss

class NeuralNetwork:
    def __init__(self, x, y, epoch, lossfunction="MSE"):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 3)
        self.weights2 = np.random.rand(3, 5)
        self.weights3 = np.random.rand(5, 3)
        self.weights4 = np.random.rand(3, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.epoch = epoch
        self.losslist = []
        self.lossfunction = lossfunction

    def feedforward(self, inputvalue):
        self.layer1 = sigmoid(np.dot(inputvalue, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        self.output = sigmoid(np.dot(self.layer3, self.weights4))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # if(self.lossfunction == "MSE"):
        dLossbydy = MSErrDeri(self.y,self.output)
        # elif(self.lossfunction == "HingeLoss"):
            # dLossbydy = hingeLossDeri(self.y,self.output,self.input)
        d_weights4 = np.dot(
            self.layer3.T, (dLossbydy * sigmoid_derivative(self.output)))

        d_weights3 = np.dot(self.layer2.T,  (np.dot(dLossbydy * sigmoid_derivative(
            self.output), self.weights4.T) * sigmoid_derivative(self.layer3)))

        d_weights2 = np.dot(self.layer1.T,  (np.dot((np.dot(dLossbydy * sigmoid_derivative(
            self.output), self.weights4.T) * sigmoid_derivative(self.layer3)), self.weights3.T) * sigmoid_derivative(self.layer2)))

        d_weights1 = np.dot(self.input.T,  (np.dot((np.dot((np.dot(dLossbydy * sigmoid_derivative(
            self.output), self.weights4.T) * sigmoid_derivative(self.layer3)), self.weights3.T) * sigmoid_derivative(self.layer2)), self.weights2.T)
            * sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function

        self.weights4 += d_weights4
        self.weights3 += d_weights3
        self.weights2 += d_weights2
        self.weights1 += d_weights1

    def train(self, showgraph=True):
        for i in range(self.epoch):
            nn.feedforward(self.input)
            nn.backprop()
            self.losslist.append(self.calculateLoss())

        if(showgraph):
            plt.plot(range(self.epoch), self.losslist,
                        c="red", label="Loss")
            plt.legend()
            plt.title("Loss vs Iteration by using loss function : {}".format(self.lossfunction))
            plt.show()

    def calculateLoss(self):
        loss = 0
        for i in range(len(self.input)):
            nn.feedforward(self.input[i])
            if(self.lossfunction == 'MSE'):
                loss = MSEerr(self.y,self.output)
            elif(self.lossfunction == 'HingeLoss'):
                loss = hinge_loss(self.y,self.output)
        print('Loss is ', loss)
        return loss

    def accuracy(self, pred, true):
        c = 0
        for i in range(true.shape[0]):
            if(pred[i] == true[i]):
                c += 1
        return c

    def calculateAccuracy(self, predictedout, actualout):
        acc = self.accuracy(np.round(predictedout), actualout)
        acc = acc/actualout.shape[0]
        return acc

    def test(self, testx, testy):
        self.feedforward(testx)
        print(self.calculateAccuracy(self.output, testy))


if __name__ == "__main__":
    def create_data(samplesize, mu1, cov1, mu2, cov2):
        class_zeros = np.random.multivariate_normal(mu1, cov1, samplesize)
        class_ones = np.random.multivariate_normal(mu2, cov2, samplesize)
        return class_zeros, class_ones

    # Creating Data...........................
    samplesize = 50
    mu1 = np.array([2, 3])
    mu2 = np.array([0, 0])
    cov1 = np.array([[10, 0.01], [0.01, 10]])
    cov2 = np.array([[1, 0], [0, 3]])

    class0y = np.zeros(samplesize)
    class1y = np.ones(samplesize)
    class_zeros, class_ones = create_data(samplesize, mu1, cov1, mu2, cov2)
    plt.scatter(class_zeros.T[0], class_zeros.T[1], c="orange", label="Class0")
    plt.scatter(class_ones.T[0], class_ones.T[1], c="blue", label="Class1")
    plt.legend()
    plt.show()

    # Splitting data into train and test............................
    # and randomizing
    class0trainsample, class0testsample, class0ytrain, class0ytest = train_test_split(
        class_zeros, class0y, test_size=0.2, random_state=0)
    class1trainsample, class1testsample, class1ytrain, class1ytest, = train_test_split(
        class_ones, class1y, test_size=0.2, random_state=0)

    trainsamples = np.concatenate((class0trainsample, class1trainsample))
    trainyclassification = np.concatenate((class0ytrain, class1ytrain))
    testsamples = np.concatenate((class0testsample, class1testsample))
    testyclassification = np.concatenate((class0ytest, class1ytest))
    train_data = list(zip(trainsamples, trainyclassification))
    np.random.shuffle(train_data)
    trainsamples, trainyclassification = zip(*train_data)
    test_data = list(zip(testsamples, testyclassification))
    np.random.shuffle(test_data)
    testsamples, testyclassification = zip(*test_data)

    trainsamples = np.array(trainsamples)
    trainyclassification = np.c_[trainyclassification].T
    testsamples = np.array(testsamples)
    testyclassification = np.c_[testyclassification].T

    ones = np.ones((trainsamples.shape[0], trainsamples.shape[1]+1))
    ones[:, :-1] = trainsamples
    trainsamples = ones

    ones = np.ones((testsamples.shape[0], testsamples.shape[1]+1))
    ones[:, :-1] = testsamples
    testsamples = ones

    epoch = 500
    # Random Weights
    nn = NeuralNetwork(trainsamples, trainyclassification, epoch)
    nn.train()
    nn.test(testsamples, testyclassification)
    nn.test(trainsamples, trainyclassification)

    nn = NeuralNetwork(trainsamples, trainyclassification, epoch, "HingeLoss")
    nn.train()
    nn.test(testsamples, testyclassification)
    nn.test(trainsamples, trainyclassification)
