from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self):
        self.losses=[]
        self.train_accuracies=[]
    def fit(self,x,y,epochs):
        x=self.transform_x(x)
        y=self.transform_y(y)

        self.weights=np.zeros(x.shape[1])
        self.bias=0

        for i in range(epochs):
            x_dot_weights=np.matmul(x.transpose(),self.weights)+self.bias
            pred=self.sigmoid(x_dot_weights)
            error_w,error_b=self.compute_gradients(x,y,pred)

    def compute_gradients(self,x,y,pred):
        gradients_w=np.matmul(x.transpose(),pred-y)
        gradients_b=np.mean(pred-y)
        gradients_w=np.array([np.mean(i) for i in gradients_w])
        return gradients_w,gradients_b
    def update_parameters(self,error_w,error_b):
        self.weights=self.weights-0.1*error_w
        self.bias=self.bias-0.1*error_b        



    def sigmoid(self,x):
        return np.array([self.sigmoid_function(i) for i in x])
    
    def sigmoid_function(self,x):
        if x>=0:
            return 1/1+np.exp(-x)
        else:
            return (np.exp(-x))/1+np.exp(-x)
    def transform_x(self,x):
        x=deepcopy(x)
        return x.values
    def transform_y(self,y):
        y=deepcopy(y)
        return y.values.reshape(y.shape[0],1)
