#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the training data

dfx = pd.read_csv('linearX.csv')
dfy = pd.read_csv('linearY.csv')

dfx = dfx.values #it changes dataframe to numpy array
dfy = dfy.values
print(dfx.shape)
#intially it is in (99,1) shape
x = dfx.reshape((-1,)) #reshaping for making it a single row numpy array
y = dfy.reshape((-1,))
print(x.shape, y.shape)

print(x)

plt.scatter(x,y)
x = (x-x.mean())/x.std() #Normalisation
y=y #y is already properly normalised
plt.scatter(x,y)
plt.show()

def hypothesis(x,theta): #hx
    return theta[0] + theta[1]*x # c + mx

def error(x,y,theta): #J(theta)
    error = 0
    for i in range(x.shape[0]): #in matrix of x x.shape[0] iterates data point of dataset (rows) ##doubt : we have already flattened so why
        hx = hypothesis(x[i], theta)
        error += (hx - y[i])**2
    return error

def gradient(x,y,theta):

    grad = np.zeros((2,)) # [0. , 0.]
    for i in range(x.shape[0]): # for every data point
        hx = hypothesis(x[i], theta) 
        grad[0] += (hx-y[i]) #derivative of error function with respect to intercept (del J/del c) 
        grad[1] += (hx-y[i])*x[i] # derivative of error function with respect to slope (del J/del m) 

    return grad

def gradientDescent(x,y,learning_rate = 0.001): #alpha or eta is learning rate

    # random theta for testing (#testing or training ???????? DOUBT)
    theta = np.array([-2.0,0.0]) # theta[0] or 'c' = -2.0 and theta[1] or 'm' = 0.0

    max_iteration = 100 #after this gradientDescent will break
    itr = 0

    error_list = [] #Doubt what is the use of list here???
    theta_list = []
    while(itr<=max_iteration):
        grad = gradient(x,y,theta)
        err = error(x,y,theta)
        error_list.append(err)
        theta_list.append(theta)
        # Learning Rule
        theta[0] -= learning_rate*grad[0] #update rule of 'c'
        theta[1] -= learning_rate*grad[1] #update rule of 'm'

        itr += 1

    return theta, error_list, theta_list

final_theta, error_list, theta_list = gradientDescent(x,y)

plt.plot(error_list)
plt.show()

print(final_theta)

# plot the line for testing data

xtest = np.linspace(-2,6,10)
print(xtest)

plt.scatter(x,y,label = 'Training Data')
plt.plot(xtest, hypothesis(xtest, final_theta), color='orange', label='prediction')
plt.legend()
plt.show()
