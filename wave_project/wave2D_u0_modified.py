#!/usr/bin/env python
"""
2D wave equation with variable wave velocity,
solved by finite differences.

Original code: Hans Petter Langtangen (not much left of it...)
"""
import time, sys
try:
    from scitools.std import linspace, newaxis, zeros, sqrt, exp, meshgrid, pi, cos, log, movie
except:
    print "Scitools not installed, exiting.."; sys.exit(1)

def solver(I, V_, f_, c, Lx, Ly, Nx, Ny, dt, T, b,
           user_action=None, version='scalar', skip_every_n_frame=10, show_cpu_time=False,display_warnings=True, plotting=False):

    order = 'C' # Store arrays in a column-major order (in memory)

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xv = x[:,newaxis]          # for vectorized function evaluations
    yv = y[newaxis,:]

    # Assuming c is a function:
    c_ = zeros((Nx+1,Ny+1), order='c')
    for i,xx in enumerate(x):
        for j,yy in enumerate(y):
            c_[i,j] = c(xx,yy)  # Loop through x and y with indices i,j at the same time
    c_max = c_.max()            # Pick out the largest value from c(x,y)
    q = c_**2

    stability_limit = (1/float(c_max))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # shortcut for max time step is to use i.e. dt = -1
        safety_factor = -dt    # use negative dt as safety factor
        extra_factor  = 1      # Easy way to make dt even smaller
        dt = safety_factor*stability_limit*extra_factor
    elif dt > stability_limit and display_warnings:
        print '\nWarning: (Unless you are testing the program), be aware that'
        print 'dt: %g is currently exceeding the stability limit: %g\n' %(dt, stability_limit)
    Nt = int(round(T/float(dt)))
    t  = linspace(0, Nt*dt, Nt+1)              # mesh points in time
    dt2 = dt**2

    # Constants for simple calculation
    A = (1 + b*dt/2)**(-1)
    B = (b*dt/2 - 1)
    dtdx2 = dt**2/(2*dx**2)
    dtdy2 = dt**2/(2*dy**2)

    # Make f(x,y,t) and V(x,y) ready for computation with different schemes
    if f_ is None or f_ == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((xv.shape[0], yv.shape[1]))
    else:
        if version == 'scalar':
            f = f_
    if V_ is None or V_ == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((xv.shape[0], yv.shape[1]))
    else:
        if version == 'scalar':
            V = V_

    if version == 'vectorized': # Generate and fill matrices for first timestep
        f = zeros((Nx+1,Ny+1), order=order)
        V = zeros((Nx+1,Ny+1), order=order)
        f[:,:] = f_(xv,yv,0)
        V[:,:] = V_(xv,yv)

    u   = zeros((Nx+1,Ny+1), order=order)   # solution array
    u_1 = zeros((Nx+1,Ny+1), order=order)   # solution at t-dt
    u_2 = zeros((Nx+1,Ny+1), order=order)   # solution at t-2*dt

    Ix = range(0, u.shape[0])               # Index set notation
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])

    import time; t0 = time.clock()          # for measuring CPU time

    # Load initial condition into u_1
    if version == 'scalar':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
    else: # use vectorized version
        u_1[:,:] = I(xv, yv)

    if user_action is not None:
        if plotting:
            user_action(u_1, x, xv, y, yv, t, 0, skip_every_n_frame)
        else:
            user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u,cpu_time = advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B,
                            dt2, dtdx2,dtdy2, V, step1=True)
    else:
        u,cpu_time = advance_vectorized(u, u_1, u_2, q, f, t, n, A, B,
                            dt2, dtdx2,dtdy2, V, step1=True)

    if user_action is not None:
        if plotting:
            user_action(u, x, xv, y, yv, t, 1, skip_every_n_frame)
        else:
            user_action(u_1, x, xv, y, yv, t, 1)

    # Update data structures for next step
    u_2, u_1, u = u_1, u, u_2

    # Time loop for all later steps
    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u,cpu_time = advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B, dt2, dtdx2,dtdy2)
            if show_cpu_time:
                percent = (float(n)/It[-2])*100.0
                sys.stdout.write("\rLast step took: %.3f sec with [scalar-code]. Computation is %d%% " %(cpu_time,percent))
                sys.stdout.flush()
        else: # Use vectorized code
            f[:,:] = f_(xv, yv, t[n])  # must precompute the matrix f
            u,cpu_time = advance_vectorized(u, u_1, u_2, q, f, t, n, A, B,
                                dt2, dtdx2,dtdy2)
            if show_cpu_time:
                percent = (float(n)/It[-2])*100.0
                sys.stdout.write("\rLast step took: %.5f sec with [vec-code]. Computation is %d%% " %(cpu_time,percent))
                sys.stdout.flush()

        if user_action is not None:
            if plotting:
                if user_action(u, x, xv, y, yv, t, n+1, skip_every_n_frame):
                    break
            else:
                if user_action(u, x, xv, y, yv, t, n+1):
                    break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to set u = u_1 if u is to be returned!
    t1 = time.clock()
    # dt might be computed in this function so return the value
    return dt, t1 - t0


def advance_vectorized(u, u_1, u_2, q, f, t, n, A, B,
                    dt2, dtdx2,dtdy2, V=None, step1=False):
    """ Haakon (me) code  """
    t1 = time.clock()
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    if step1:
        I = u_1; dt = sqrt(dt2)
        u[1:-1,1:-1] = 0.5*(2*I[1:-1,1:-1] - 2*B*dt*V[1:-1,1:-1] + dt2*f[1:-1,1:-1]   \
               + dtdx2*( (q[1:-1,1:-1] + q[2:,1:-1])  * (I[2:,1:-1]   - I[1:-1,1:-1]) \
               -         (q[1:-1,1:-1] + q[:-2,1:-1]) * (I[1:-1,1:-1] - I[:-2,1:-1])) \
               + dtdy2*( (q[1:-1,1:-1] + q[1:-1,2:])  * (I[1:-1,2:]   - I[1:-1,1:-1]) \
               -         (q[1:-1,1:-1] + q[1:-1,:-2]) * (I[1:-1,1:-1] - I[1:-1,:-2])))
    else:
        u[1:-1,1:-1] = A*( 2*u_1[1:-1,1:-1] + B*u_2[1:-1,1:-1] + dt2*f[1:-1,1:-1]         \
               + dtdx2*( (q[1:-1,1:-1] + q[2:,1:-1])  * (u_1[2:,1:-1]   - u_1[1:-1,1:-1]) \
               -         (q[1:-1,1:-1] + q[:-2,1:-1]) * (u_1[1:-1,1:-1] - u_1[:-2,1:-1])) \
               + dtdy2*( (q[1:-1,1:-1] + q[1:-1,2:])  * (u_1[1:-1,2:]   - u_1[1:-1,1:-1]) \
               -         (q[1:-1,1:-1] + q[1:-1,:-2]) * (u_1[1:-1,1:-1] - u_1[1:-1,:-2])))

    ######################################
    # Neumann boundary condition du/dn=0 #
    ######################################
    if step1:
        i = Ix[0] # 1) Boundary where x = 0
        u[i,1:-1] = 0.5*(2*I[i,1:-1] - 2*B*dt*V[i,1:-1] + dt2*f[i,1:-1]        \
               + dtdx2*2*(q[i,1:-1] + q[i+1,1:-1]) * (I[i+1,1:-1] - I[i,1:-1]) \
               + dtdy2*( (q[i,1:-1] + q[i,2:])     * (I[i,2:]     - I[i,1:-1]) \
               -         (q[i,1:-1] + q[i,:-2])    * (I[i,1:-1]   - I[i,:-2])))

        i = Ix[-1] # 1) Boundary where x = Nx
        u[i,1:-1] = 0.5*(2*I[i,1:-1] - 2*B*dt*V[i,1:-1] + dt2*f[i,1:-1]        \
               + dtdx2*2*(q[i,1:-1] + q[i-1,1:-1]) * (I[i-1,1:-1] - I[i,1:-1]) \
               + dtdy2*( (q[i,1:-1] + q[i,2:])     * (I[i,2:]     - I[i,1:-1]) \
               -         (q[i,1:-1] + q[i,:-2])    * (I[i,1:-1]   - I[i,:-2])))

        j = Iy[0] # 1) Boundary where y = 0
        u[1:-1,j] = 0.5*(2*I[1:-1,j] - 2*B*dt*V[1:-1,j] + dt2*f[1:-1,j]          \
               + dtdx2*( (q[1:-1,j] + q[2:,j])      * (I[2:,j]      - I[1:-1,j]) \
               -         (q[1:-1,j] + q[:-2,j])     * (I[1:-1,j]    - I[:-2,j])) \
               + dtdy2*2*(q[1:-1,j] + q[1:-1:,j+1]) * (I[1:-1:,j+1] - I[1:-1,j]))

        j = Iy[-1] # 1) Boundary where y = Ny
        u[1:-1,j] = 0.5*(2*I[1:-1,j] - 2*B*dt*V[1:-1,j] + dt2*f[1:-1,j]          \
               + dtdx2*( (q[1:-1,j] + q[2:,j])      * (I[2:,j]      - I[1:-1,j]) \
               -         (q[1:-1,j] + q[:-2,j])     * (I[1:-1,j]    - I[:-2,j])) \
               + dtdy2*2*(q[1:-1,j] + q[1:-1:,j-1]) * (I[1:-1:,j-1] - I[1:-1,j]))

        # Special formula for the four corner points
        i = Ix[0]; j = Iy[0]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V[i,j] + dt2*f[i,j]        \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j]) \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j]))

        i = Ix[0]; j = Iy[-1]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V[i,j] + dt2*f[i,j]        \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j]) \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (I[i,j-1] - I[i,j]))

        i = Ix[-1]; j = Iy[0]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V[i,j] + dt2*f[i,j]        \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (I[i-1,j] - I[i,j]) \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j]))

        i = Ix[-1]; j = Iy[-1]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V[i,j] + dt2*f[i,j]        \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (I[i-1,j] - I[i,j]) \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (I[i,j-1] - I[i,j]))

    else: # Any step NOT first
        i = Ix[0] # 1) Boundary where x = 0
        u[i,1:-1] = A*( 2*u_1[i,1:-1] + B*u_2[i,1:-1] + dt2*f[i,1:-1]                 \
             + dtdx2*2*(q[i,1:-1] + q[i+1,1:-1]) * (u_1[i+1,1:-1] - u_1[i,1:-1])  \
             + dtdy2*( (q[i,1:-1] + q[i,2:])      * (u_1[i,2:]      - u_1[i,1:-1])  \
             -         (q[i,1:-1] + q[i,:-2])     * (u_1[i,1:-1]    - u_1[i,:-2])))

        i = Ix[-1] # 1) Boundary where x = Nx
        u[i,1:-1] = A*( 2*u_1[i,1:-1] + B*u_2[i,1:-1] + dt2*f[i,1:-1]                 \
             + dtdx2*2*(q[i,1:-1] + q[i-1,1:-1]) * (u_1[i-1,1:-1] - u_1[i,1:-1])  \
             + dtdy2*( (q[i,1:-1] + q[i,2:])      * (u_1[i,2:]      - u_1[i,1:-1])  \
             -         (q[i,1:-1] + q[i,:-2])     * (u_1[i,1:-1]    - u_1[i,:-2])))

        j = Iy[0] # 1) Boundary where y = 0
        u[1:-1,j] = A*( 2*u_1[1:-1,j] + B*u_2[1:-1,j] + dt2*f[1:-1,j]               \
               + dtdx2*( (q[1:-1,j] + q[2:,j])     * (u_1[2:,j]     - u_1[1:-1,j])  \
               -         (q[1:-1,j] + q[:-2,j])    * (u_1[1:-1,j]   - u_1[:-2,j]))  \
               + dtdy2*2*(q[1:-1,j] + q[1:-1,j+1]) * (u_1[1:-1,j+1] - u_1[1:-1,j]) )

        j = Iy[-1] # 1) Boundary where y = Ny
        u[1:-1,j] = A*( 2*u_1[1:-1,j] + B*u_2[1:-1,j] + dt2*f[1:-1,j]               \
               + dtdx2*( (q[1:-1,j] + q[2:,j])     * (u_1[2:,j]     - u_1[1:-1,j])  \
               -         (q[1:-1,j] + q[:-2,j])    * (u_1[1:-1,j]   - u_1[:-2,j]))  \
               + dtdy2*2*(q[1:-1,j] + q[1:-1,j-1]) * (u_1[1:-1,j-1] - u_1[1:-1,j]) )

        # Special formula for the four corner points
        i = Ix[0]; j = Iy[0]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f[i,j]  \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j]))

        i = Ix[0]; j = Iy[-1]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f[i,j]  \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (u_1[i,j-1] - u_1[i,j]))

        i = Ix[-1]; j = Iy[0]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f[i,j]  \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (u_1[i-1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j]))

        i = Ix[-1]; j = Iy[-1]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f[i,j]  \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (u_1[i-1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (u_1[i,j-1] - u_1[i,j]))
    CPU_time = time.clock() - t1
    return u,CPU_time


def advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B, dt2, dtdx2,dtdy2,
                   V=None, step1=False):
    t1 = time.clock()
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    if step1: # Special formula for step 1
        I = u_1; dt = sqrt(dt2)
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
                       + dtdx2*( (q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
                       -         (q[i,j] + q[i-1,j]) * (I[i,j]   - I[i-1,j]))       \
                       + dtdy2*( (q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j])          \
                       -         (q[i,j] + q[i,j-1]) * (I[i,j]   - I[i,j-1])))
    else: # Compute ALL interior points
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])    \
                       + dtdx2*( (q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])    \
                       -         (q[i,j] + q[i-1,j]) * (u_1[i,j]   - u_1[i-1,j])) \
                       + dtdy2*( (q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j])    \
                       -         (q[i,j] + q[i,j-1]) * (u_1[i,j]   - u_1[i,j-1])))

    # Neumann boundary condition du/dx = 0
    if step1:
        i = Ix[0] # 1) Boundary where x = 0
        for j in Iy[1:-1]:
            u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
                   + dtdx2*2*(q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
                   + dtdy2*( (q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j])          \
                   -         (q[i,j] + q[i,j-1]) * (I[i,j]   - I[i,j-1])))

        i = Ix[-1] # 1) Boundary where x = Nx
        for j in Iy[1:-1]:
            u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
                   + dtdx2*2*(q[i,j] + q[i-1,j]) * (I[i-1,j] - I[i,j])          \
                   + dtdy2*( (q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j])          \
                   -         (q[i,j] + q[i,j-1]) * (I[i,j]   - I[i,j-1])))

        j = Iy[0] # 1) Boundary where y = 0
        for i in Ix[1:-1]:
            u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
                   + dtdx2*( (q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
                   -         (q[i,j] + q[i-1,j]) * (I[i,j]   - I[i-1,j]))       \
                   + dtdy2*2*(q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j]))

        j = Iy[-1] # 1) Boundary where y = Ny
        for i in Ix[1:-1]:
            u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
                   + dtdx2*( (q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
                   -         (q[i,j] + q[i-1,j]) * (I[i,j]   - I[i-1,j]))       \
                   + dtdy2*2*(q[i,j] + q[i,j-1]) * (I[i,j-1] - I[i,j]))

        # Special formula for the four corner points
        i = Ix[0]; j = Iy[0]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j]))

        i = Ix[0]; j = Iy[-1]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (I[i+1,j] - I[i,j])          \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (I[i,j-1] - I[i,j]))

        i = Ix[-1]; j = Iy[0]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (I[i-1,j] - I[i,j])          \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (I[i,j+1] - I[i,j]))

        i = Ix[-1]; j = Iy[-1]
        u[i,j] = 0.5*(2*I[i,j] - 2*B*dt*V(x[i],y[j]) + dt2*f(x[i], y[j], 0) \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (I[i-1,j] - I[i,j])          \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (I[i,j-1] - I[i,j]))

    else: # Any step NOT first
        i = Ix[0] # 1) Boundary where x = 0
        for j in Iy[1:-1]:
            u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
                   + dtdx2*2*(q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])  \
                   + dtdy2*( (q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j])  \
                   -         (q[i,j] + q[i,j-1]) * (u_1[i,j]   - u_1[i,j-1])))

        i = Ix[-1] # 1) Boundary where x = Nx
        for j in Iy[1:-1]:
            u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
                   + dtdx2*2*(q[i,j] + q[i-1,j]) * (u_1[i-1,j] - u_1[i,j])  \
                   + dtdy2*( (q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j])  \
                   -         (q[i,j] + q[i,j-1]) * (u_1[i,j]   - u_1[i,j-1])))

        j = Iy[0] # 1) Boundary where y = 0
        for i in Ix[1:-1]:
            u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])    \
                   + dtdx2*( (q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])    \
                   -         (q[i,j] + q[i-1,j]) * (u_1[i,j]   - u_1[i-1,j])) \
                   + dtdy2*2*(q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j]) )

        j = Iy[-1] # 1) Boundary where y = Ny
        for i in Ix[1:-1]:
            u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])    \
                   + dtdx2*( (q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])    \
                   -         (q[i,j] + q[i-1,j]) * (u_1[i,j]   - u_1[i-1,j])) \
                   + dtdy2*2*(q[i,j] + q[i,j-1]) * (u_1[i,j-1] - u_1[i,j]) )

        # Special formula for the four corner points
        i = Ix[0]; j = Iy[0]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j]))

        i = Ix[0]; j = Iy[-1]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
               + dtdx2*2*(q[i,j] + q[i+1,j]) * (u_1[i+1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (u_1[i,j-1] - u_1[i,j]))

        i = Ix[-1]; j = Iy[0]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (u_1[i-1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j+1]) * (u_1[i,j+1] - u_1[i,j]))

        i = Ix[-1]; j = Iy[-1]
        u[i,j] = A*( 2*u_1[i,j] + B*u_2[i,j] + dt2*f(x[i], y[j], t[n])  \
               + dtdx2*2*(q[i,j] + q[i-1,j]) * (u_1[i-1,j] - u_1[i,j])  \
               + dtdy2*2*(q[i,j] + q[i,j-1]) * (u_1[i,j-1] - u_1[i,j]))
    CPU_time = time.clock() - t1
    return u, CPU_time
