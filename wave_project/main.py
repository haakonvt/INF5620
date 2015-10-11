from wave2D_u0_modified import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2D_wave(u, x, xv, y, yv, t, n):
    if n == 0: # First step
        global X,Y
        X,Y = meshgrid(xv,yv)
    fig = plt.figure()
    ax  = Axes3D(fig)
    ax.set_zlim3d(-0.10, 0.3)
    ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=plt.cm.gist_earth)
    plt.savefig('2d_wave_%.4i.png' %n)
    plt.show()
    plt.close() # Close to save memory (and not get 1000 windows open..)

c = lambda x,y: 0.8 # Testing constant wave velocity
f = None
V = None

def I(x,y): # Initial distr. = symmetric bell curve
    return exp(-80*(x-0.5)**2)*exp(-80*(y-0.5)**2)

Lx =  1;  Ly = 1
Nx =  40; Ny = 40
dt = -1 # Shortcut for maximum timestep
T  =  5 # Number of seconds to run
b  =  1

# Call solver-function
solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, \
       user_action=plot_2D_wave, version='scalar')
