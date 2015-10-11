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


if __name__ == '__main__':
    test_constant()
