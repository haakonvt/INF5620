import numpy as np
from matplotlib import pyplot as plt
from wave1D_u0 import solver

def viz(I, V, f, c, L, dt, C, T, umin, umax, animate=True):
    """Run solver and visualize u at each time level."""
    import scitools.std as plt
    import time, glob, os

    def plot_u(u, x, t, n):
        """user_action function for solver."""
        u_e = u_exact(x,t,n)
        plt.plot(x, u, 'r-', x, u_e, 'bo',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], legend=['numerical sol.', 'Exact sol.'],
                 show=True)

        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig('frame_%04d.png' % n)  # for movie making

    def write_error(u,x,t,n):
        u_e      = u_exact(x,t,n)
        max_diff = ((u-u_e)).max()
        err_list.append(max_diff)
        print "At t= %.2f, maximum error: %.4E" %(t[n],max_diff)

        axis_max = 0.00073*len(t) # Experimentally found (max_err scales ~linearly)
        plt.plot(x, u-u_e, 'r-',
                 xlabel='x', ylabel='u_{num}-u_{exact}',
                 axis=[0, (len(x)-1)/2.0, -axis_max, axis_max],
                 title='t=%f' % t[n], legend='Absolute error',
                 show=True)
    def plot_both(u,x,t,n):
        u_e      = u_exact(x,t,n)
        max_diff = ((u-u_e)).max()
        err_list.append(max_diff)
        print "At t= %.2f, maximum error: %.4E" %(t[n],max_diff)

        axis_max = 0.00073*len(t) # Experimentally found (max_err scales ~linearly)
        plt.subplot(211)
        plt.plot(x, u, 'r-', x, u_e, 'bo',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], legend=['numerical sol.', 'Exact sol.'])
        plt.subplot(212)
        plt.plot(x, u-u_e, 'r-',
                 xlabel='x', ylabel='u_{num}-u_{exact}',
                 axis=[0, (len(x)-1)/2.0, -axis_max, axis_max],
                 legend='Absolute error',
                 show=True)


    # Clean up old movie frames
    for filename in glob.glob('frame_*.png'):
        os.remove(filename)

    # Choose user_action
    if animate == None:
        user_action = plot_both
    else:
        user_action = plot_u if animate else write_error
    # Let the magic happen
    u, x, t, cpu = solver(I, V, f, c, L, dt, C, T, user_action)


def u_exact(x,t,n):
        t    = t[n]
        u_e  = A*np.sin(pi*m*x/L)*np.cos(pi*m*c*t/L)
        return u_e

# Simulate a standing wave
pi = np.pi

#Initial conditions
# task a) lambda = 2L/m, P = 2L/(c*m)
A = 1.0
m = 3.0
L = 12.0
c = 2.0
T = 10.0*2*L/(m*c)

I = lambda x: A*np.sin(pi*m*x/L)          # u(x,t=0)
V = lambda x: (A*pi*m/L)*np.cos(pi*m*x/L) # v(x,t=0)
f = 0                                     # Source term

Nx = 80    # Grid in spatial dimension
dt = 0.2  # Timestep
C  = 0.8   # Courant number

umin = 1.2*A
umax = -umin
global err_list
err_list = []

# True = plot solution vs exact. False = plot error. None = plot both
user_action = None
cpu = viz(I,0,f,c,L,dt,C,T,umin,umax,user_action)

if user_action == False:
    plt.figure()
    plt.plot(err_list)
    plt.show()
raw_input('Please press enter to finish')
