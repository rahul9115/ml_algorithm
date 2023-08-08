import numpy as np
import random
def cluster_points(X,mu):
    clusters={}
    for x in X:
        bestmukey=min([(i[0],np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)],key=lambda x:x[1])
        try:
            clusters[bestmukey].append(x)
        except:
            clusters[bestmukey]=[x]
    return clusters
    
def reevaluate_centers(clusters,mu):
    newmu=[]
    keys=sorted(clusters.keys())
    for i in keys:
        newmu.append(np.mean(clusters[i],axis=0))
    return newmu

    
def has_converged(old_mu,mu):
    return set([tuple(i) for i in old_mu])==set([tuple(i) for i in mu])

def find_centers(X,k):
    old_mu=random.sample(X,k)
    print(old_mu)
    mu=random.sample(X,k)
    while not has_converged(old_mu,mu):
        old_mu=mu
        clusters=cluster_points(X,mu)
        mu=reevaluate_centers(old_mu,clusters)

