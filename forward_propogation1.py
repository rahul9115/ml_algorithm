import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import imageio
from sklearn.model_selection import train_test_split
from IPython.display import HTML

data,labels=make_moons(n_samples=200,noise=0.04,random_state=np.random.seed(0))
color_map=matplotlib.colors.LinearSegmentedColormap.from_list("",["red","blue"])
plt.scatter(data[:,0],data[:,1],c=labels,cmap=color_map)
plt.show()
X_train,X_test,y_train,y_test=train_test_split(data,labels,stratify=labels,random_state=0)

class FeedForwardNetwork:
    def __init__(self) -> None:
        self.w1=np.random.rand(2,2)
        self.w2=np.random.rand(2,1)
        
        self.b1=0
        self.b2=0
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def forward_propogation(self,x):
        
        self.a1=np.matmul(x,self.w1)+self.b1
        self.h1=self.sigmoid(self.a1)
        self.a2=np.matmul(self.h1,self.w2)+self.b2
        self.h2=self.sigmoid(self.a2)
        return self.h2
ffn_v = FeedForwardNetwork()
forward_matrices=list(ffn_v.forward_propogation(X_train).transpose())

def plot_heat_map(observation):
    fig=plt.figure(figsize=(10,1))
    sns.heatmap(forward_matrices[observation],annot=True,cmap=color_map,vmin=4,vmax=4)
    plt.title("Observation"+str(observation))
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] +(3,))
    return image
imageio.mimsave("./forwardpropagation_viz.gif", [plot_heat_map(i) for i in range(0,len(forward_matrices),len(forward_matrices)//15)], fps=1)

    
    
        
