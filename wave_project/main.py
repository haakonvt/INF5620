import matplotlib
matplotlib.use('Agg')   # Make the plotting invisible, see .png-files!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wave2D_u0_modified import *
import os, glob # To remove files

def plot_2D_wave(u, x, xv, y, yv, t, n, skip_every_n_frame=10):
    if n == 0: # First step, make meshgrid
        global X,Y,count,dpi_setting;
        dpi_setting = 120 # Default is 100
        X,Y   = meshgrid(xv,yv)
        count = skip_every_n_frame # Make sure to plot initial state
    if count == skip_every_n_frame:
        fig = plt.figure()
        ax  = Axes3D(fig)
        ax.set_zlim3d(-0.10, 0.3)
        ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=plt.cm.gist_earth)
        plt.savefig('2d_wave_%.4i.png' %n, dpi = dpi_setting, antialiased=False)
        plt.close() # Close to save memory (and not get 1000 windows open..)
        count = 0   # Reset counter
    else:
        count += 1

def test_case():
    Lx =  1;  Ly = 1
    Nx =  80; Ny = 80
    dt = -1     # Shortcut for maximum timestep
    T  =  4   # Number of seconds to run
    b  =  1.  # Damping term, if larger than zero

    scalar_or_vec = 'vectorized'
    #scalar_or_vec = 'scalar'

    # Call solver-function
    solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, \
           user_action=plot_2D_wave, version=scalar_or_vec, show_cpu_time=True)
    def c(x,y):
        # Half the domain got different velocity
        if x<0.5:
            return 0.3
        else:
            return 0.9
        #return 0.8 # Testing constant wave velocity

    def f(x,y,t):
        return 0

    def V(x,y):
        return 0

    def I(x,y): # Initial distr.
        """A = 0.5*exp(-80*(x-1)**2)*exp(-80*(y-1)**2)
        B = 0.5*exp(-80*(x-1)**2)*exp(-80*(y)**2)
        C = 0.5*exp(-80*(x)**2)*exp(-80*(y-1)**2)
        D = 0.5*exp(-80*(x)**2)*exp(-80*(y)**2)
        return A + B + C + D"""
        return 0.5*exp(-80*(x-1)**2)

def test_standing_undamped_waves(A=0.2, mx=3, my=2, Lx=1, Ly=1):
    kx      = mx*pi/Lx; ky = my*pi/Ly
    w       = 0.5*pi/(Lx+Ly)*sqrt(mx**2 + my**2) # See report for derivation

    ue = lambda x,y  : A*cos(kx*x)*cos(ky*y)*cos(w*t)
    I  = lambda x,y  : A*cos(kx*x)*cos(ky*y)
    V  = lambda x,y  : 0
    f  = lambda x,y,t: 0
    c  = lambda x,y  : 1

    solver(I, V, f, c, Lx, Ly, Nx=80, Ny=80, dt=-1, T=0.55, b=0,
           user_action=plot_2D_wave, version='vectorized', skip_every_n_frame=1, show_cpu_time=True)


if __name__ == '__main__':
    test_standing_undamped_waves()
    print '\nRun the following commands to create a movie-gif-file and remove all plot files:'
    print '>> convert -delay 5 "2d_wave_*.png" "movie.gif"'
    print '>> rm 2d_wave_*.png'

    keep_frames = raw_input('Press enter to delete all frames! (or enter "keep" to keep)')
    if not keep_frames == 'keep':
        for frame in glob.glob("2d_wave_*.png"): # Find all frames
            os.remove(frame)                     # Delete these frames
