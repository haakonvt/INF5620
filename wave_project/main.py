import matplotlib
matplotlib.use('Agg')   # Make the plotting invisible
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wave2D_u0_modified import *
import os, glob # To remove files


def plot_2D_wave(u, x, xv, y, yv, t, n, skip_every_n_frame=10):
    if n == 0: # First step, make meshgrid
        global X,Y,count,dpi_setting;
        dpi_setting = 150 # Default is 100
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


def plot_gauss_drop_corner():
    """
    This is just a cool test-case, if you want a cool-looking gif-animation
    of a gaussian drop in the corner.
    Not relevant for the tasks in the Wave Project
    """
    Lx =  1;  Ly = 1
    Nx =  60; Ny = 60
    dt = -1   # Shortcut for maximum timestep
    T  =  6   # Number of seconds to run
    b  =  1.  # Damping term, if larger than zero

    scalar_or_vec = 'vectorized'
    #scalar_or_vec = 'scalar'

    def c(x,y):
        # Half the domain got different velocity
        """if x<0.5:
            return 0.3
        else:
            return 0.9"""
        return 0.3 # Testing constant wave velocity

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
        return 0.8*exp(-80*(x-1)**2)*exp(-80*(y-1)**2)

    # Call solver-function
    solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, \
           user_action=plot_2D_wave, version=scalar_or_vec, \
           skip_every_n_frame=3, show_cpu_time=True, plotting=True)




if __name__ == '__main__':
    print """\n
    This is just a cool test-case, if you want a cool-looking gif-animation,
    of a gaussian drop in the corner. Not relevant for the tasks in the
    Wave Project. Run test-file instead with:
    >>> nosetests tests.py
    """
    for frame in glob.glob("2d_wave_*.png"): # Is there old frames?
        os.remove(frame)                     # Delete these frames!

    plot_gauss_drop_corner() # Test-case

    print "\n\nConverting all frames to a gif... Takes some time"
    movie('2d_wave_*.png', encoder='convert', output_file='gauss_wave2D.gif'
            , delay=5, quiet=True)

    keep_frames = raw_input('\nMovie has been made!\nPress enter to delete all frames! (or enter "keep" to keep)')
    if not keep_frames == 'keep':
        for frame in glob.glob("2d_wave_*.png"): # Find all frames
            os.remove(frame)                     # Delete these frames
