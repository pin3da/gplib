import numpy as np
import pylab as pb

pb.ion()


def m_load(fname) :
    return fromfile(fname, sep='\n')

def plotModel(x,m,v):
    #input: x, np.array with d columns representing m prediction points
    # m, predicted mean at x, np.array of shape (m,1)
    # v, predicted variance matrix, np.array of shape (m,m)
    x = x.flatten()
    m = m.flatten()
    v = np.diag(v)
    upper=m+2*np.sqrt(v)
    lower=m-2*np.sqrt(v)
    pb.plot(x,m,color="#204a87",linewidth=2)
    pb.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color="#729fcf",alpha=0.3)
    pb.plot(x,upper,color="#204a87",linewidth=0.2)
    pb.plot(x,lower,color="#204a87",linewidth=0.2)
    # pb.plot(xpoints,ypoints,'rx')

def plot(x, y):
    pb.plot(x, y)

#X = np.loadtxt("x.mio")
#Y = np.loadtxt("y.mio")
nX = np.loadtxt("nx.mio")
nY = np.loadtxt("pre.mio")
cov = np.loadtxt("cov.mio")

# plotModel(nX, nY, cov)
