import numpy as np

import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation
    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
    def margin(self,training_inputs):
        margin = 99999999
        for inputs in training_inputs:
            wtx = np.abs(np.dot(inputs, self.weights[1:]) + self.weights[0])
            # print('dot proudct', wtx)
            if(margin> wtx):
                # print('changing margin')
                margin = wtx
        print('Margin', margin)


training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)
perceptron.margin(training_inputs)
print('perceptron weights:::', perceptron.weights)



from sklearn.linear_model import LogisticRegression
#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(training_inputs, labels)
print('logmodel coef',logmodel.coef_) # returns a matrix of weights (coefficients)
