#!/usr/bin/env python
"""
1D wave equation with Dirichlet or Neumann conditions
and variable wave velocity::
 u, x, t, cpu = solver(I, V, f, c, U_0, U_L, L, dt, C, T,
                       user_action=None, version='scalar',
                       stability_safety_factor=1.0)
Solve the wave equation u_tt = (c**2*u_x)_x + f(x,t) on (0,L) with
u=U_0 or du/dn=0 on x=0, and u=u_L or du/dn=0
on x = L. If U_0 or U_L equals None, the du/dn=0 condition
is used, otherwise U_0(t) and/or U_L(t) are used for Dirichlet cond.
Initial conditions: u=I(x), u_t=V(x).
T is the stop time for the simulation.
dt is the desired time step.
C is the Courant number (=max(c)*dt/dx).
stability_safety_factor enters the stability criterion:
C <= stability_safety_factor (<=1).
I, f, U_0, U_L, and c are functions: I(x), f(x,t), U_0(t),
U_L(t), c(x).
U_0 and U_L can also be 0, or None, where None implies
du/dn=0 boundary condition. f and V can also be 0 or None
(equivalent to 0). c can be a number or a function c(x).
user_action is a function of (u, x, t, n) where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""
import time, glob, shutil, os
import numpy as np

def solver(I, V, f, c, U_0, U_L, L, dt, C, T,
           user_action=None, version='scalar',
           stability_safety_factor=1.0,use_std_neuman_bcs=True):
    """Solve u_tt=(c^2*u_x)_x + f on (0,L)x(0,T]."""
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)      # Mesh points in time

    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 101)])
    dx = dt*c_max/(stability_safety_factor*C)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)          # Mesh points in space

    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    # Wrap user-given f, I, V, U_0, U_L if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0

    # Make hash of all input data
    """import hashlib, inspect
    data = inspect.getsource(I) + '_' + inspect.getsource(V) + \
           '_' + inspect.getsource(f) + '_' + str(c) + '_' + \
           ('None' if U_0 is None else inspect.getsource(U_0)) + \
           ('None' if U_L is None else inspect.getsource(U_L)) + \
           '_' + str(L) + str(dt) + '_' + str(C) + '_' + str(T) + \
           '_' + str(stability_safety_factor)
    hashed_input = hashlib.sha1(data).hexdigest()
    if os.path.isfile('.' + hashed_input + '_archive.npz'):
        # Simulation is already run
        return -1, hashed_input"""

    u   = np.zeros(Nx+1)   # Solution array at new time level
    u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

    import time;  t0 = time.clock()  # CPU time measurement

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    if use_std_neuman_bcs == True:
        i = Ix[0]
        if U_0 is None:
            # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
            # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
            ip1 = i+1
            im1 = ip1  # i-1 -> i+1
            u[i] = u_1[i] + dt*V(x[i]) + \
                   0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                           0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
            0.5*dt2*f(x[i], t[0])
        else:
            u[i] = U_0(dt)

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1  # i+1 -> i-1
            u[i] = u_1[i] + dt*V(x[i]) + \
                   0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                           0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
            0.5*dt2*f(x[i], t[0])
        else:
            u[i] = U_L(dt)
    else: # Use other discretization for Neumann bcs.:
        i = Ix[0]
        if U_0 is None:
            # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
            # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
            ip1 = i+1
            im1 = ip1  # i-1 -> i+1
            u[i] = u_1[i] + dt*V(x[i]) + 0.5*dt2*f(x[i], t[0]) + \
                   C2*q[i]*(u_1[ip1] - u_1[i])
        else:
            u[i] = U_0(dt)

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1  # i+1 -> i-1
            u[i] = u_1[i] + dt*V(x[i]) + 0.5*dt2*f(x[i], t[0]) + \
                   C2*2*q[i]*(u_1[im1] - u_1[i])
        else:
            u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        if version == 'scalar':
            for i in Ix[1:-1]:
                u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - \
                        0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
                dt2*f(x[i], t[n])

        elif version == 'vectorized':
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
            C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
                0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + \
            dt2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        if use_std_neuman_bcs == True:
            i = Ix[0]
            if U_0 is None:
                # Set boundary values
                # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
                # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
                ip1 = i+1
                im1 = ip1
                u[i] = - u_2[i] + 2*u_1[i] + \
                       C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                           0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
                dt2*f(x[i], t[n])
            else:
                u[i] = U_0(t[n+1])

            i = Ix[-1]
            if U_L is None:
                im1 = i-1
                ip1 = im1
                u[i] = - u_2[i] + 2*u_1[i] + \
                       C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                           0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
                dt2*f(x[i], t[n])
            else:
                u[i] = U_L(t[n+1])
        else:
            """ Use the approximation (54) in lecture notes:
                http://hplgit.github.io/num-methods-for-PDEs/doc/pub/wave/html/._wave003.html#mjx-eqn-54"""
            i = Ix[0]
            if U_0 is None:
                # Set boundary values
                # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
                # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
                ip1 = i+1
                im1 = ip1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) + \
                       C2*2*q[i]*(u_1[ip1]-u_1[i])
            else:
                u[i] = U_0(t[n+1])

            i = Ix[-1]
            if U_L is None:
                im1 = i-1
                ip1 = im1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) + \
                       C2*2*q[i]*(u_1[im1] - u_1[i])
            else:
                u[i] = U_L(t[n+1])

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = t0 - time.clock()
    return cpu_time#, hashed_input

class PlotAndStoreSolution:
    """
    Class for the user_action function in solver.
    Visualizes the solution only.
    """
    def __init__(
        self,
        casename='tmp',    # Prefix in filenames
        umin=-1, umax=1,   # Fixed range of y axis
        pause_between_frames=None,  # Movie speed
        backend='matplotlib',       # or 'gnuplot' or None
        screen_movie=True, # Show movie on screen?
        title='',          # Extra message in title
        skip_frame=1,      # Skip every skip_frame frame
        filename=None):    # Name of file with solutions
        self.casename = casename
        self.yaxis = [umin, umax]
        self.pause = pause_between_frames
        self.backend = backend
        self.E_list = []
        if backend is None:
            # Use native matplotlib
            import matplotlib.pyplot as plt
        elif backend in ('matplotlib', 'gnuplot'):
            module = 'scitools.easyviz.' + backend + '_'
            exec('import %s as plt' % module)
        self.plt = plt
        self.screen_movie = screen_movie
        self.title = title
        self.skip_frame = skip_frame
        self.filename = filename
        if filename is not None:
            # Store time points when u is written to file
            self.t = []
            filenames = glob.glob('.' + self.filename + '*.dat.npz')
            for filename in filenames:
                os.remove(filename)

        # Clean up old movie frames
        for filename in glob.glob('frame_*.png'):
            os.remove(filename)

    def __call__(self, u, x, t, n):
        """
        Callback function user_action, call by solver:
        Store solution, plot on screen and save to file.
        """
        # Save solution u to a file using numpy.savez
        if self.filename is not None:
            name = 'u%04d' % n  # array name
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            savez(fname, **kwargs)
            self.t.append(t[n])  # store corresponding time value
            if n == 0:           # save x once
                savez('.' + self.filename + '_x.dat', x=x)

        # Animate
        if n % self.skip_frame != 0:
            return
        title = 't=%.3f' % t[n]
        if self.title:
            title = self.title + ' ' + title

        L = 1.0
        u_e = np.cos(np.pi*x/L)*np.cos(t[n])

        if self.backend is None:
            # native matplotlib animation
            if n == 0:
                self.plt.ion()
                self.lines = self.plt.plot(x, u, 'r-', x, u_e, 'b-')
                self.plt.axis([x[0], x[-1],
                               self.yaxis[0], self.yaxis[1]])
                self.plt.xlabel('x')
                self.plt.ylabel('u')
                self.plt.title(title)
                self.plt.legend(['t=%.3f' % t[n]])
            else:
                # Update new solution
                self.lines[0].set_ydata(u)
                self.plt.legend(['t=%.3f' % t[n]])
                self.plt.draw()
        else:
            # scitools.easyviz animation
            self.plt.plot(x, u, 'r-', x, u_e, 'b-',
                          xlabel='x', ylabel='u',
                          axis=[x[0], x[-1],
                                self.yaxis[0], self.yaxis[1]],
                          title=title,
                          show=self.screen_movie)
        # pause
        if t[n] == 0:
            time.sleep(1)  # let initial condition stay 1 s
        else:
            if self.pause is None:
                pause = 0.2 if u.size < 100 else 0
            time.sleep(pause)

        # Error analysis
        dt = t[1]-t[0]
        e  = u_e - u
        E = np.sqrt(dt*sum(e**2))
        self.E_list.append(E)
        #print n,len(t)
        if n + self.skip_frame > len(t)-1:
            self.plt.figure()
            self.plt.plot(self.E_list)
            print "Timestep dt:",dt
            print "Maximum error in this run:", max(self.E_list)
            raw_input('Press enter to kill graphics...')

        #self.plt.savefig('frame_%04d.png' % (n))

    def make_movie_file(self):
        """
        Create subdirectory based on casename, move all plot
        frame files to this directory, and generate
        an index.html for viewing the movie in a browser
        (as a sequence of PNG files).
        """
        # Make HTML movie in a subdirectory
        directory = self.casename
        if os.path.isdir(directory):
            shutil.rmtree(directory)   # rm -rf directory
        os.mkdir(directory)            # mkdir directory
        # mv frame_*.png directory
        for filename in glob.glob('frame_*.png'):
            os.rename(filename, os.path.join(directory, filename))
        os.chdir(directory)        # cd directory
        fps = 4 # frames per second
        if self.backend is not None:
            from scitools.std import movie
            movie('frame_*.png', encoder='html',
                  output_file='index.html', fps=fps)

        # Make other movie formats: Flash, Webm, Ogg, MP4
        codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                         libtheora='ogg')
        filespec = 'frame_%04d.png'
        movie_program = 'ffmpeg'  # or 'avconv'
        for codec in codec2ext:
            ext = codec2ext[codec]
            cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
                  '-vcodec %(codec)s movie.%(ext)s' % vars()
            os.system(cmd)
        os.chdir(os.pardir)  # move back to parent directory

    def close_file(self, hashed_input):
        """
        Merge all files from savez calls into one archive.
        hashed_input is a string reflecting input data
        for this simulation (made by solver).
        """
        if self.filename is not None:
            # Save all the time points where solutions are saved
            savez('.' + self.filename + '_t.dat',
                  t=array(self.t, dtype=float))

            # Merge all savez files to one zip archive
            archive_name = '.' + hashed_input + '_archive.npz'
            filenames = glob.glob('.' + self.filename + '*.dat.npz')
            merge_zip_archives(filenames, archive_name)
	    print 'Archive name:', archive_name
            # data = numpy.load(archive); data.files holds names
            # data[name] extract the array

class PlotMediumAndSolution(PlotAndStoreSolution):
    def __init__(self, medium, **kwargs):
        """Mark medium in plot: medium=[x_L, x_R]."""
        self.medium = medium
        PlotAndStoreSolution.__init__(self, **kwargs)

    def __call__(self, u, x, t, n):
        # Save solution u to a file using numpy.savez
        if self.filename is not None:
            name = 'u%04d' % n  # array name
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            savez(fname, **kwargs)
            self.t.append(t[n])  # store corresponding time value
            if n == 0:           # save x once
                savez('.' + self.filename + '_x.dat', x=x)

        # Animate
        if n % self.skip_frame != 0:
            return
        # Plot u and mark medium x=x_L and x=x_R
        x_L, x_R = self.medium
        umin, umax = self.yaxis
        title = 'Nx=%d' % (x.size-1)
        if self.title:
            title = self.title + ' ' + title
        if self.backend is None:
            # native matplotlib animation
            if n == 0:
                self.plt.ion()
                self.lines = self.plt.plot(
                    x, u, 'r-',
                    [x_L, x_L], [umin, umax], 'k--',
                    [x_R, x_R], [umin, umax], 'k--')
                self.plt.axis([x[0], x[-1],
                               self.yaxis[0], self.yaxis[1]])
                self.plt.xlabel('x')
                self.plt.ylabel('u')
                self.plt.title(title)
                self.plt.text(0.75, 1.0, 'C=0.25')
                self.plt.text(0.32, 1.0, 'C=1')
                self.plt.legend(['t=%.3f' % t[n]])
            else:
                # Update new solution
                self.lines[0].set_ydata(u)
                self.plt.legend(['t=%.3f' % t[n]])
                self.plt.draw()
        else:
            # scitools.easyviz animation
            self.plt.plot(x, u, 'r-',
                          [x_L, x_L], [umin, umax], 'k--',
                          [x_R, x_R], [umin, umax], 'k--',
                          xlabel='x', ylabel='u',
                          axis=[x[0], x[-1],
                                self.yaxis[0], self.yaxis[1]],
                          title=title,
                          show=self.screen_movie)
        # pause
        if t[n] == 0:
            time.sleep(2)  # let initial condition stay 2 s
        else:
            if self.pause is None:
                pause = 0.2 if u.size < 100 else 0
            time.sleep(pause)

        #self.plt.savefig('frame_%04d.png' % (n))

if __name__ == '__main__':
    pass
