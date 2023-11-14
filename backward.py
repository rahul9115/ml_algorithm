import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons,load_iris
import matplotlib.colors

data=load_iris()
X=data.data
y=data.target
data,labels=make_moons(n_samples=200,noise=0.04,random_state=np.random.seed(0))
color_map=matplotlib.colors.LinearSegmentedColormap.from_list("",["red","blue"])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=color_map)
plt.show()
X_train,y_train,X_test,y_test=train_test_split(data,labels,stratify=labels)
print(X_train.shape)

class Neural_Networks:
    def __init__(self):
        self.w1=np.random.rand(2,2)
        self.w2=np.random.rand(2,2)
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def backward_propogation(self,X):
        global y_train
        #forward propogation
        Z1=np.matmul(X,self.w1)
        A1=self.sigmoid(Z1)
        Z2=np.matmul(A1,self.w2)
        A2=self.sigmoid(Z2)

        #backward propogation
        l=[]
        for i in y_train:
            l.append([i])
        y_train=np.array(l)

        error1=A2-y_train
        dw1=error1*A2*(1-A2)

        error2=np.dot(dw1,self.w2.T)
        dw2=error2*A1*(1-A1)

        w1_update=np.dot(A1.T,dw1)
        w2_update=np.dot(X.T,dw2)

        self.w1=self.w1-0.1*w1_update
        self.w2=self.w2-0.1*w2_update





