from scitools.std import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def z1(x,y):
    w = 2; a = w*0.1
    if sqrt((x-0.5)**2 + (y-0.5)**2) <= a:
        B = 0.2*(1+cos(pi*(x-0.5)/a))*(1+cos(pi*(y-0.5)/a))
    else:
        B = 0
    return B

def z2(x,y):
    B = 0.6*exp(-30*(x-0.5)**2)*exp(-30*(y-0.5)**2)
    return B

def z3(x,y):
    if (0.3 < x < 0.7) and (0.25 < y < 0.75):
        B = 0.6
    else:
        B = 0
    return B

N = 101

x = linspace(0,1,N)
y = linspace(0,1,N)

def c(x,y,n):
    if n == 1:
        return sqrt(z1(x,y))
    if n == 2:
        return sqrt(z2(x,y))
    if n == 3:
        return sqrt(z3(x,y))

for n in [1,2,3]:
    c_ = zeros((N,N))
    for i,xx in enumerate(x):
        for j,yy in enumerate(y):
            c_[i,j] = c(xx,yy,n)


    xv  = x[:,newaxis]          # for vectorized function evaluations
    yv  = y[newaxis,:]
    X,Y = meshgrid(xv,yv)
    fig = plt.figure()
    ax  = Axes3D(fig)
    ax.set_zlim3d(0, 2)
    ax.plot_surface(X, Y, c_, rstride=1, cstride=1, cmap=plt.cm.gist_earth)
    plt.savefig('2d_wave_%.4i.png' %n, dpi = 150, antialiased=False)
