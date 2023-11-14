import numpy as np
class Sigmoid:
     def sigmoid(self,x):
        
        return 1/(1+np.exp(-x))
s=Sigmoid()
print(s.sigmoid(np.matmul(np.array([1,2,2,2]).reshape(2,2),np.random.rand(2,2))))


