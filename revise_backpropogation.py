import numpy as np
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,make_moons
import matplotlib.pyplot as plt
'''
This stratify parameter makes a split so that the proportion of 
values in the sample produced will be the same as the proportion 
of values provided to parameter stratify.
For example, if variable y is a binary categorical variable with 
values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y 
will make sure that your random split has 25% of 0's and 75% of 1's.


'''
data,labels=make_moons(n_samples=200,noise=0.04,random_state=np.random.seed(0))
color_map=matplotlib.colors.LinearSegmentedColormap("",["red","blue"])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=color_map)
X_train,x_test,y_train,y_test=train_test_split(data,labels,stratify=labels)
class Neural_Networks:
    def __init__(self):
        self.w1=np.random.rand(2,2)
        self.w2=np.random.rand(2,1)
        self.N=len(y_train)
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def back_propogation(self,X):
        global y_train
        #forward propogation
        l=[]
        for i in y_train:
            l.append([i])
        y_train=np.array(l)
        for i in range(1000):
            Z1=np.matmul(X,self.w1)
            A1=self.sigmoid(Z1)
            Z2=np.matmul(A1,self.w2)
            A2=self.sigmoid(Z2)
            
            
            error1=A2-y_train
            print("this",error1.shape)
            dw1=error1*A2*(1-A2)
            print("DW1",dw1.shape)
            error2=np.dot(dw1,self.w2.T)
            dw2=error2*A1*(1-A1)
            print("DW2",dw2.shape)
            w2_update=np.dot(A1.T,dw1)
            w1_update=np.dot(X.T,dw2)
            print(w1_update.shape)
            self.w1=self.w1-0.1*w1_update
            self.w2=self.w2-0.1*w2_update
            print("values",self.w1,self.w2)
nn=Neural_Networks()
nn.back_propogation(X_train)     



                    
