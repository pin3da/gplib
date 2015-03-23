import numpy as np
import pylab as pb

# Graphic stuff
pb.ion()

def plotModel(x,m,v,xpoints,ypoints):
    #input:     x, np.array with d columns representing m prediction points
    #           m, predicted mean at x, np.array of shape (m,1)
    #           v, predicted variance matrix, np.array of shape (m,m)
    x = x.flatten()
    m = m.flatten()
    v = np.diag(v)
    upper=m+2*np.sqrt(v)
    lower=m-2*np.sqrt(v)
    pb.plot(x,m,color="#204a87",linewidth=2)
    pb.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color="#729fcf",alpha=0.3)
    pb.plot(x,upper,color="#204a87",linewidth=0.2)
    pb.plot(x,lower,color="#204a87",linewidth=0.2)
    pb.plot(xpoints,ypoints,'rx')


def gaussian(X,Y,sigma2=1.5,theta=4.0):
    d2 = np.sum((X[:,None,:] - Y[None,:,:]) ** 2 / theta ** 2, 2)
    k = sigma2*np.exp(-d2 / 2.0)
    return (k + np.eye(X.shape[0]) * 1e-9)


# X =  np.random.uniform(0.0, 3.0, (4, 3))
X = np.array([[0.04912227, 2.67544358, 2.53769772],
              [1.64872528, 0.52406991, 1.35924321],
              [1.52015109, 0.72726333, 0.06925668],
              [0.60605489, 0.74525666, 1.97343891]])
result =  np.array(gaussian(X, X))

