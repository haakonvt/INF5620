import numpy as np
from matplotlib import pyplot as plt
from wave1D_u0 import solver,viz

def viz(I, V, f, c, L, dt, C, T, umin, umax, animate=True):
    """Run solver and visualize u at each time level."""
    import scitools.std as plt
    import time, glob, os

    def plot_u(u, x, t, n):
        """user_action function for solver."""
        u_e = u_exact(x,t,n)
        plt.plot(x, u, 'r-', x, u_e, 'b-',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], show=True)
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig('frame_%04d.png' % n)  # for movie making

    def write_error(u,x,t,n):
        u_e      = u_exact(x,t,n)
        max_diff = ((u-u_e)).max()
        err_list.append(max_diff)
        #print "Timestep %i, maximum erroer: %.4E" %(t[n],max_diff)

    # Clean up old movie frames
    for filename in glob.glob('frame_*.png'):
        os.remove(filename)

    user_action = plot_u if animate else write_error
    u, x, t, cpu = solver(I, V, f, c, L, dt, C, T, user_action)

    # Make movie files
    """fps = 4  # Frames per second
    plt.movie('frame_*.png', encoder='html', fps=fps,
              output_file='movie.html')
    codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                     libtheora='ogg')
    filespec = 'frame_%04d.png'
    movie_program = 'avconv'  # or 'ffmpeg'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
              '-vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)"""


def u_exact(x,t,n):
        t    = t[n]
        u_e  = A*np.sin(pi*m*x/L)*np.cos(pi*m*c*t/L)
        return u_e

# Simulate a standing wave
pi = np.pi

#Initial conditions
A = 1.0
m = 9.0
L = 12.0
c = 2.0
T = 10.0*2*pi/(pi*m*c/L)

I = lambda x: A*np.sin(pi*m*x/L)          # u(x,t=0)
V = lambda x: (A*pi*m/L)*np.cos(pi*m*x/L) # v(x,t=0)
f = 0                                     # Source term

Nx = 80    # Grid in spatial dimension
dt = 0.01  # Timestep
C  = 0.8   # Courant number

umin = 1.2*A
umax = -umin
global err_list
err_list = []
cpu = viz(I,0,f,c,L,dt,C,T,umin,umax,animate=False)

plt.plot(err_list)
plt.show()
raw_input('Please press enter to finish')
