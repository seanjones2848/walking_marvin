import time, math, random, bisect, copy
import numpy as np
import gym
from scipy import optimize as opt

env = gym.make('Marvin-v0')

class Neural_Network(object):
    def __init__(self):
        #defining Hyperparameters
        self.inputLayerSize = 24
        self.outputLayerSize = 4
        self.hiddenLayerSize = 8
        #define weights and biases
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, solf.outputLayerSize)
    def forward(self, X):
        #propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    def sigmoidPrime(self, z):
        #Gradient of Sigmoid
        return np.exp(-z)/((1 + np.exp(-z))**2)
    def cost(self, X, y):
        #Compute cost of given X, y using W1 and W2 in NN currently
        return (0.5*sum((y-self.forward(X)**2)))
	def costPrime(self, X, y):
        #Compute partial derivatives with respect to W1 and W2 for a given X, y
		delta3 = np.multiply(-(y-self.forward(X)), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2
    def getParams(self):
        #Get W1 and W2 unrolled into vector
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))
    def setParams(self, params):
        #Set W1 and W2 using single vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.reshape(params[W1_end:W2_end], (self.heddenLayerSize, self.ouputLayerSize))
    def gradients(self, X, y):
        dW1, dW2 = self.costPrime(X, y)
        return np.concatenate((dW1.ravel(), dW2.ravel()))
class trainer(object):
    def __init__(self, N):
        #local reference to NN:
        self.N = N
    def callback(self, params):
        self.N.setParams(params)
        self.J.append(self.N.cost(self.X, self.y))
    def costWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.cost(X, y)
        grad = self.N.gradients(X, y)
        return cost, grad
    def train(self, X, y):
        #make internal variables for callback
        self.X = X
        self.y = y
        self.J = []
        params = self.N.getParams()
        options = {'maxiter':200, 'disp':True}
        optimized = opt.minimize(self.costWrapper, params, jac=True, method='BFGS', \
                         args=(X, y), options=options, callback=self.callback)
        self.N.setParams(optimized.x)
        self.optimization = optimized
