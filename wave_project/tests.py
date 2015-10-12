from main import plot_2D_wave
from wave2D_u0_modified import *

def constant_solution(Nx, Ny, version):

    exact_solution = lambda x, y, t: 3
    I = lambda x, y   : 3
    V = lambda x, y   : 0
    f = lambda x, y, t: 0
    c = lambda x, y   : 1

    Lx = 7.0; Ly = 7.0
    b  = 0.5; dt = -1 # -1 is shortcut to max stable dt
    T  = 10

    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e  = exact_solution(xv, yv, t[n])
        diff = abs(u -u_e).max()
        E.append(diff)
        tol  = 1E-12
        msg  = "diff=%g, step=%d, time=%g" %(diff, n, t[n])
        assert diff < tol, msg

    dt, cpu = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b,
        user_action=assert_no_error, version=version, show_cpu_time=False)
    return dt, cpu

def test_constant():
    # Test a series of meshes where Nx > Ny and Nx < Ny
    global E
    versions = 'scalar', 'vectorized'
    for Nx in range(2, 6, 2):
        for Ny in range(2, 6, 2):
            for ver in versions:
                E = [] # Will contain error from computation
                print 'testing', ver, 'for %dx%d mesh' % (Nx, Ny)
                constant_solution(Nx, Ny, ver)
                print '- largest error:', max(E)

class LastTimestep:
    """Need class to save values (also after the call has been made)"""
    def __call__(self, u, x, xv, y, yv, t, n):
        last_timestep = len(t)-1
        if n == last_timestep:
            self.x = x.copy(); self.y = y.copy()
            self.u = u.copy(); self.t = t[n]

def test_plug():
    """Check that an initial plug is correct back after one period."""
    V = lambda x, y   : 0
    f = lambda x, y, t: 0
    c = lambda x, y   : 1

    Lx = 1.1; Ly = 1.1
    Nx = 11;  Ny = 11
    dt = 0.1;  T = 1.1
    b  = 0 # No damping term!

    def Ixs(x,y): # For scalar scheme
        if abs(x-Lx/2.0) > 0.1:
            return 0
        else:
            return 0.2

    def Ixv(x,y):
        I = zeros(x.shape)
        for i in range(len(x[:,0])):
            if abs(x[i,0]-Lx/2.0) > 0.1:
                I[i,0] = 0
            else:
                I[i,0] = 0.2
        return I

    def Iys(x,y):
        if abs(y-Ly/2.0) > 0.1:
            return 0
        else:
            return 0.2

    def Iyv(x,y):
        I = zeros(y.shape)
        for j in range(len(y[0,:])):
            if abs(y[0,j]-Ly/2.0) > 0.1:
                I[0,j] = 0
            else:
                I[0,j] = 0.2
        return I

    # Make user action an LastTimestep-instance
    lasttimestep = LastTimestep()

    # Check I_scalar and I_vectorized in both x- and y-direction:
    for Is,Iv in [(Ixs,Ixv), (Iys,Iyv)]:
        # Test plug wave in x/y-direction with scalar and vectorized code
        solver(Is, V, f, c, Lx, Ly, Nx, Ny, dt, T, b,
            user_action=lasttimestep, version='scalar',skip_every_n_frame=0, display_warnings=False)
        u_scalar = lasttimestep.u # Store last u
        #solver(Is, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, user_action=plot_2D_wave, version='scalar')
        #raw_input('check frames and enter...')
        solver(Iv, V, f, c, Lx, Ly, Nx, Ny, dt, T, b,
            user_action=lasttimestep, version='vectorized',skip_every_n_frame=0, display_warnings=False)
        u_vec = lasttimestep.u
        #solver(Iv, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, user_action=plot_2D_wave, version='vectorized')

        diff = abs(u_scalar - u_vec).max()
        tol = 1E-15
        assert diff < tol

        u_0 = zeros((Nx+1,Ny+1))
        u_0[:,:] = Iv(lasttimestep.x[:,newaxis], lasttimestep.y[newaxis,:])
        diff1 = abs(u_scalar - u_0).max()
        diff2 = abs(u_vec    - u_0).max() # This is basicly already tested, 'u_vec_x = u_scalar_x'
        assert diff1 < tol and diff2 < tol

if __name__ == '__main__':
    print "Run tests with:\n>>> nosetests tests.py"
    raw_input('Press enter to cnontinue...')
    test_constant()
    test_plug()
