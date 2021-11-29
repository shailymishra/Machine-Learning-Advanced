import numpy as np
import matplotlib.pyplot as plt

###########################################################3
### ALL COST FUNCTION
# Huber Cost Function
def huber(targets, predictions, delta):
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        error = np.where(np.abs(target - prediction) < delta , 0.5*((target - prediction)**2), delta*np.abs(target - prediction) - 0.5*(delta**2))
        accumulated_error += error
    huber_error = accumulated_error
    return huber_error
# Log Cosh loss
import sympy as sym
def logcosh(targets, predictions):
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        x = prediction - target
        error = sym.log(sym.cosh(x))
        accumulated_error += (error)
    logcosh_error = accumulated_error
    return logcosh_error
# Quantile Reg
def quantileReg(targets, predictions, gamma):
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        x = prediction - target
        error = sym.log(sym.cosh(x))
        error = np.where(target >= prediction, gamma*(np.abs(target-prediction)), (1-gamma)*(np.abs(target-prediction)))
        accumulated_error += (error)
    quantile_error = accumulated_error
    return quantile_error
# MSE
def mse(predictions, targets):
    samples_num = len(predictions)
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
    return mae_error
# MAE
def mae(predictions, targets):
    samples_num = len(predictions)
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += np.abs(prediction - target)
    mae_error = (1.0 / samples_num) * accumulated_error
    return mae_error
####################################################################################3
def predict(x,parameters):
    return x*parameters['w'][0] + parameters['b']

##########################################
####   1. PLOT THE DATA ##################
#######################################

slope = 4
intercept = 3
noOfSample = 1000
low = -3000
high = 3000
mu = 0
sigma = 1
X = np.random.randint(low,high ,size=noOfSample)
X = np.sort(X)
noise = np.random.normal(mu, sigma, noOfSample)
y = slope*X + intercept + noise

plt.scatter(X,y, label= "Data Points for Line with slope 4 and intercept 3")
plt.title("Data Samples")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()


#####################################################################################################
############# 2. PLOT THE LOSS FUNCTION  keeping intercept constant = 3 #############################
#####################################################################################################

w_space = np.arange(-3., 10, 1.0)
parameters_list = list()
for w_value in w_space:
    parameters = {'w': np.array([w_value]), 'b': 0}
    parameters_list.append(parameters)
    
mae_list = list()
mse_list = list()
huber_list_1 = list()
huber_list_2 = list()
huber_list_3 = list()
huber_list = list()
logcosh_list = list()
quantile_list_1 = list()


for parameter_set in parameters_list:
    mae_error = mae([predict(x, parameter_set) for x in X], y)
    mae_list.append(mae_error)
    
    mse_error = mse([predict(x, parameter_set) for x in X], y)
    mse_list.append(mse_error)

    huber_error = huber([predict(x, parameter_set) for x in X], y,100)
    huber_list_1.append(huber_error)

    huber_error = huber([predict(x, parameter_set) for x in X], y,1000)
    huber_list_2.append(huber_error)

    huber_error = huber([predict(x, parameter_set) for x in X], y,3000)
    huber_list_3.append(huber_error)

    logcosh_error = logcosh([predict(x, parameter_set) for x in X], y)
    logcosh_list.append(logcosh_error)

    quantile_error = quantileReg([predict(x, parameter_set) for x in X], y,0.25)
    quantile_list_1.append(quantile_error)    


def setGraph(x_values,y_values, graph,title,color,legendstring='', xlabel="w", ylabel="Loss"):
    graph.scatter(x_values, y_values, edgecolor='black', linewidth='1', s=35, zorder=2, c=color)
    graph.plot(x_values, y_values, zorder=1,c=color, label=legendstring)
    graph.set_title(title)
    graph.legend()
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.set_xticks(np.arange(-3., 10, 1.0))
    graph.grid(color='blue', linestyle='--', linewidth=1, alpha=0.1)
    graph.spines["top"].set_visible(False)
    graph.spines["right"].set_visible(False)
    graph.spines["bottom"].set_visible(False)
    graph.spines["left"].set_visible(False)

ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
setGraph(w_space, mse_list,ax1,'MSE (Original Line : y=4x+3) ', 'blue','', 'w (Slope) - intercept b=0 ')
setGraph(w_space,mae_list,ax2,'MAE (Original Line : y=4x+3) ', 'orange','','w (Slope) - intercept b=0 ')
setGraph(w_space,huber_list_1,ax3,'Huber (Original Line : y=4x+3)', 'green', 'Delta =100','w (Slope) - intercept b=0 ')
setGraph(w_space,huber_list_2,ax3,'Huber (Original Line : y=4x+3)', 'blue', 'Delta = 1000','w (Slope) - intercept b=0 ')
setGraph(w_space,huber_list_3,ax3,'Huber (Original Line : y=4x+3)', 'orange', 'Delta = 3000','w (Slope) - intercept b=0 ')
setGraph(w_space,logcosh_list,ax4,'LogCosh (Original Line : y=4x+3)', 'magenta','w (Slope) - intercept b=0 ')
setGraph(w_space,quantile_list_1,ax5,'Qunatile (Original Line : y=4x+3)', 'yellow','Gamma = 0.25','w (Slope) - intercept b=0 ')
plt.show()


############################################################################################
###################    Learning Process - Gradient Descent ################################
##########################################################################################

def CalculateParametersThroughHuber(m,b,X,Y,delta, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        if abs(Y[i] - m*X[i] - b) <= delta:
            m_deriv += -X[i] * (Y[i] - (m*X[i] + b))
            b_deriv += - (Y[i] - (m*X[i] + b))
        else:
            m_deriv += delta * X[i] * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
            b_deriv += delta * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])    
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate
    return m, b

def CalculateParametersThroughLogCosh(m,b,X,Y, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        m_deriv += X[i] * sym.tanh((m*X[i] + b)-Y[i] )
        b_deriv += sym.tanh((m*X[i] + b)-Y[i])
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate
    return m, b

def CalculateParametersThroughQunatile(m,b,X,Y,gamma, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        if (Y[i] - m*X[i] - b) < 0:
            m_deriv += -X[i] * (gamma - 1) * (-1)
            b_deriv += -1 * (gamma - 1) * (-1)
        else:
            m_deriv += -X[i] * gamma * (1) 
            b_deriv +=  -1 * gamma * (1)    
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate
    return m, b


def gradientDescentOnHuber(learning_rate,maxiteration = 1000,delta=500, w=-3, b=3):
    startParameters = {'w' : [w], 'b':b}
    errors = []
    allw = []
    currentParameters = startParameters
    iteration = 0
    while(iteration<maxiteration):
        w,b = CalculateParametersThroughHuber(currentParameters['w'][0], currentParameters['b'],X,y,delta,learning_rate)
        print('W is', w, 'b::', b)
        currentParameters = {'w' : [w], 'b':b }
        currentError = huber([predict(x, currentParameters) for x in X], y,delta)
        errors.append(currentError)
        allw.append(w)
        iteration = iteration + 1
    print('Final W is', w , 'Final b is ', b)
    iterations = np.arange(0,maxiteration+1)
    return errors,iterations,allw


def gradientDescentOnLogCosh(learning_rate,maxiteration = 100,w=-3,b=3):
    startParameters = {'w' : [w], 'b':b}
    errors = []
    allw = []
    currentParameters = startParameters
    iteration = 0
    while(iteration<maxiteration):
        w,b = CalculateParametersThroughLogCosh(currentParameters['w'][0], currentParameters['b'],X,y,learning_rate)
        print('w is', w, 'b is', b)
        currentParameters = {'w' : [w], 'b':b }
        currentError = logcosh([predict(x, currentParameters) for x in X], y)
        errors.append(currentError)
        allw.append(w)
        iteration = iteration + 1
    print('Final W is', w , 'Final b is ', b)
    iterations = np.arange(0,maxiteration+1)
    return errors,iterations,allw


def gradientDescentOnQuantile(learning_rate,maxiteration = 200,gamma=0.75, w=-3, b=3):
    startParameters = {'w' : [w], 'b':b}
    errors = []
    allw = []
    currentParameters = startParameters
    iteration = 0
    while(iteration<maxiteration):
        w,b = CalculateParametersThroughQunatile(currentParameters['w'][0], currentParameters['b'],X,y,gamma,learning_rate)
        print('W is', w, 'b::', b)
        currentParameters = {'w' : [w], 'b':b }
        currentError = quantileReg([predict(x, currentParameters) for x in X], y,gamma)
        errors.append(currentError)
        allw.append(w)
        iteration = iteration + 1
    print('Final W is', w , 'Final b is ', b)
    iterations = np.arange(0,maxiteration+1)
    return errors,iterations,allw

#########    Learning  on Huber 

errors_huber,iterations_huber,allw_huber = gradientDescentOnHuber(0.000002)
errors_logcosh,iterations_logcosh,allw_logcosh = gradientDescentOnLogCosh(0.0001)
errors_quantile,iterations_quantile,allw_quantile = gradientDescentOnQuantile(0.0001)

fig, axs = plt.subplots(2, 2)
setGraph(allw_huber, errors_huber ,axs[0,0],' Huber Learning (Original Line = 4x+3) ', 'blue', 'Learning Rate = 0.00001, Delta = 500')
setGraph(allw_logcosh,errors_logcosh,axs[0,1],'LogCosh Learnig (Original Line = 4x+3)  ', 'orange', 'Learing Rate= 0.0001')
setGraph(allw_quantile,errors_quantile,axs[1,0],'Quantile Learning (Original Line = 4x+3) ', 'green', 'Learning Rate = 0.0001, Gamma =0.75')
# plt.title('Orginial line is y=4x+3')
plt.show()



