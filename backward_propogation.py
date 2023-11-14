import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.colors
from sklearn.datasets import load_iris
data = load_iris()

# Get features and target
X=data.data
y=data.target
print(y)
data,labels=make_moons(n_samples=200,noise=0.04,random_state=np.random.seed(0))
color_map=matplotlib.colors.LinearSegmentedColormap.from_list("",["red","blue"])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=color_map)
plt.show()
X_train,x_test,y_train,y_test=train_test_split(data,labels,stratify=labels)


class Neural_Networks:
    def __init__(self) -> None:
        self.w1=np.random.rand(2,2)
        self.w2=np.random.rand(2,1)
  
    def sigmoid(self,x):
        
        return 1/(1+np.exp(-x))
    def backward_propogation(self,X):
        global y_train
        print("X-Shape",X.shape)
        Z1=np.matmul(X,self.w1)
        print("new",Z1.shape)
        A1=self.sigmoid(Z1)
        print("new1",A1)
        Z2=np.matmul(A1,self.w2)
        A2=self.sigmoid(Z2)
        l=[]
        for i in y_train:
            print(i)
            l.append([i])
        y_train=np.array(l)
        print("this",y_train.shape,A2.shape)
        print("val",y_train)   
        e1=A2-y_train
        print(e1)
        dw1=e1*A2*(1-A2)
        print("DW1 shape",dw1.shape,e1.shape,A1.T.shape)
        
        e2=np.dot(dw1,self.w2.T)
        dw2=e2*A1*(1-A1)

        w2_update=np.dot(A1.T,dw1)
        w1_update=np.dot(X.T,dw2)
        print("w1",w1_update.shape,w1_update,self.w1)
        self.w1=self.w1-0.1*w1_update
        print(self.w1)
        self.w2=self.w2-0.1*w2_update
nn=Neural_Networks()
nn.backward_propogation(X_train)

        
