import numpy as np
import matplotlib.colors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  sklearn.datasets import make_moons

import seaborn as sns
import imageio
from IPython.display import HTML

data,labels=make_moons(n_samples=200,noise=0.04,random_state=np.random.seed(0))
color_map=matplotlib.colors.LinearSegmentedColormap.from_list("",["red","yellow"])
# plt.scatter(data[:,0],data[:,1],c=labels,cmap=color_map)
X_train,X_test,y_train,y_test=train_test_split(data,labels,stratify=labels,random_state=0)

class FeedForwardNetwork:
    def __init__(self) -> None:
        np.random.seed(0)
        self.w1=np.random.rand()
        self.w2=np.random.rand()
        self.w3=np.random.rand()
        self.w4=np.random.rand()
        self.w5=np.random.rand()
        self.w6=np.random.rand()
        self.b1=0
        self.b2=0
        self.b3=0
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def forward_pass(self,x):
        self.x1,self.x2=x
        self.a1=self.w1*self.x1+self.w2*self.x2+self.b1
        self.h1=self.sigmoid(self.a1)
        self.a2=self.w3*self.x1+self.w4*self.x2+self.b2
        self.h2=self.sigmoid(self.a2)
        self.a3=self.h1*self.w5+self.h2*self.w6+self.b3
        self.h3=self.sigmoid(self.a3)
        forward_matrix=np.array([[0,0,0,0,self.h3,0,0,0],
                                [0,0,self.w5*self.h1,self.w6*self.h2,self.b3,self.a3,0,0],
                                [0,0,0,self.h1,0,0,0,self.h2],
                                [self.w1*self.x1,self.w2*self.x2,self.b1,self.a1,self.w3*self.x1,self.w4*self.x2,self.b2,self.a2]])
        forward_matrices.append(forward_matrix)
forward_matrices = []
ffn = FeedForwardNetwork()
for x in X_train:
   ffn.forward_pass(x)
print(len(forward_matrices))    
def plot_heat_map(observation):
    fig=plt.figure(figsize=(10,1))
    sns.heatmap(forward_matrices[observation],annot=True,cmap=color_map,vmin=4,vmax=4)
    plt.title("Observation"+str(observation))
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] +(3,))
    return image
imageio.mimsave("./forwardpropagation_viz.gif", [plot_heat_map(i) for i in range(0,len(forward_matrices),len(forward_matrices)//15)], fps=1)


