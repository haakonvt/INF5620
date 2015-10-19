#import matplotlib
#matplotlib.use('Agg')   # Make the plotting invisible
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wave2D_u0_modified import *
import os, glob # To remove files


def plot_2D_wave(u, x, xv, y, yv, t, n, skip_every_n_frame=10):
    if n == 0: # First step, make meshgrid
        global X,Y,count,dpi_setting;
        dpi_setting = 130 # Default is 100
        X,Y   = meshgrid(xv,yv)
        count = skip_every_n_frame # Make sure to plot initial state
    if count == skip_every_n_frame:
        plt.ioff()
        fig = plt.figure()
        ax  = Axes3D(fig)
        ax.set_zlim3d(-0.10, 0.3)
        ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=plt.cm.gist_earth)
        plt.savefig('2d_wave_%.4i.png' %n, dpi = dpi_setting, antialiased=False)
        plt.close() # Close to save memory (and not get 1000 windows open..)
        count = 0   # Reset counter
    else:
        count += 1


def tsunami_case(ocean_floor_choice=1):
    """
    Runs an experiment with a chosen ocean floor hill and a tsunami over it!
    ...and of course, plots it!
    """
    ofc = ocean_floor_choice

    for frame in glob.glob("2d_wave_*.png"): # Is there old frames?
        os.remove(frame)                     # Delete these frames!

    Lx =  1;   Ly = 1
    Nx =  120; Ny = 120
    dt = -1    # Shortcut for maximum timestep
    T  =  0.4  # Number of seconds to run
    b  =  0.8  # Damping term, if larger than zero

    scalar_or_vec = 'vectorized' # We choose vectorized code of course..!

    def H(x,y,ofc):
        B0 = 1         # Constant wave speed level
        if ofc == 1:   # Gaussian addition
            B = 0.6*exp(-30*(x-0.5)**2)*exp(-30*(y-0.5)**2)
        elif ofc == 2: # Cosine hat addition
            w = 2; a = w*0.1
            if sqrt((x-0.5)**2 + (y-0.5)**2) <= a:
                B = 0.2*(1+cos(pi*(x-0.5)/a))*(1+cos(pi*(y-0.5)/a))
            else:
                B = 0
        else:          # Box addition
            if (0.3 < x < 0.7) and (0.25 < y < 0.75):
                B = 0.6
            else:
                B = 0
        return B0 - B

    def c(x,y): # c = sqrt(q)
        g      = 9.81
        c_temp = sqrt(g * H(x,y,ofc))
        return c_temp

    def I(x,y): # Initial distr.
        return 0.3*exp(-60*(x)**2)

    f = lambda x,y,t: 0 # Source term
    V = lambda x,y  : 0 # Initial du/dt

    # Call solver-function
    solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, \
           user_action=plot_2D_wave, version=scalar_or_vec, \
           skip_every_n_frame=4, show_cpu_time=True, plotting=True)



def plot_gauss_drop_corner():
    print """\n
    This is just a test-case, if you want a cool-looking gif-animation,
    of a gaussian drop in the corner. Not relevant for the tasks in the
    Wave Project. Run test-file instead with:
    >>> nosetests tests.py
    """
    for frame in glob.glob("2d_wave_*.png"): # Is there old frames?
        os.remove(frame)                     # Delete these frames!

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
    # plot_gauss_drop_corner() # Nice test-case, not part of the project

    print """\n
    Investigate a physical problem, a tsunami over a subsea hill.
    Initial distribution is a gauss-wave in x-direction.
    Choose the geometry of the ocean floor by inputting a number [1-3]:
    1: Gaussian hill
    2: "Cosine hat" hill
    3: Simple box hill\n"""
    try:
        ocean_floor_choice = int(raw_input('Input your choice: '))
        ofc = ocean_floor_choice
    except:
        print "Input must be an integer, i.e. 1,2 or 3. Try again!"
        sys.exit(1)

    if ocean_floor_choice not in [1,2,3]:
        print "Input must be an integer, i.e. 1,2 or 3. Try again!"
        sys.exit(1)

    print ""
    print "1: Gaussian hill chosen" if ofc == 1 else '2: "Cosine hat hill chosen' if ofc == 2 else "3: Simple box hill chosen"

    tsunami_case(ofc)

    print "\n\nConverting all frames to a gif... Takes some time"
    movie('2d_wave_*.png', encoder='convert', output_file='gauss_wave2D.gif'
            , fps=5, quiet=True)

    keep_frames = raw_input('\nMovie has been made!\nPress enter to delete all frames! (or enter "keep" to keep)')
    if not keep_frames == 'keep':
        for frame in glob.glob("2d_wave_*.png"): # Find all frames
            os.remove(frame)                     # Delete these frames
