#!/usr/bin/env python
"""
2D wave equation solved by finite differences

Solve the 2D wave equation u_tt = u_xx + u_yy + f(x,t) on (0,L) with
u=0 on the boundary and initial condition du/dt=0.

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f are functions: I(x,y), V(x,y), f(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.

user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""
import time, sys
try:
    from scitools.std import linspace, newaxis, zeros, sqrt, exp, meshgrid
except:
    print "Scitools not installed, exiting.."; sys.exit(1)

def solver(I, V_, f_, c, Lx, Ly, Nx, Ny, dt, T, b,
           user_action=None, version='scalar'):

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
    c_max = c_.max()            # Pick out the largest value for c(x,y)
    q = c_**2

    stability_limit = (1/float(c_max))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # shortcut for max time step is to use i.e. dt = -1
        safety_factor = -dt    # use negative dt as safety factor
        extra_factor  = 0.5    # Make dt even smaller
        dt = safety_factor*stability_limit*extra_factor
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)
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
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B,
                            dt2, dtdx2,dtdy2, V, step1=True)
    else:
        u = advance_vectorized(u, u_1, u_2, q, f, x, y, t, n, A, B,
                            dt2, dtdx2,dtdy2, V, step1=True)

    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    # Update data structures for next step
    u_2, u_1, u = u_1, u, u_2

    # Time loop for all later steps
    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B, dt2, dtdx2,dtdy2)
        else: # Use vectorized code
            f[:,:] = f_(xv, yv, t[n])  # must precompute the matrix f
            u = advance_vectorized(u, u_1, u_2, q, f, t, n, A, B,
                                dt2, dtdx2,dtdy2)

        if user_action is not None:
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
            u[i,1:-1] = 0.5*(2*I[i,1:-1] - 2*B*dt*V[i,1:-1] + dt2*f[i,1:-1]         \
                   + dtdx2*2*(q[i,1:-1] + q[i-1,1:-1]) * (I[i-1,1:-1] - I[i,1:-1])  \
                   + dtdy2*( (q[i,1:-1] + q[i,2:])     * (I[i,2:]     - I[i,1:-1])  \
                   -         (q[i,1:-1] + q[i,:-2])    * (I[i,1:-1]   - I[i,:-2])))

        j = Iy[0] # 1) Boundary where y = 0
            u[1:-1:j] = 0.5*(2*I[1:-1:j] - 2*B*dt*V[1:-1:j] + dt2*f[1:-1:j]     \
                   + dtdx2*( (q[1:-1:j] + q[2:,j]) * (I[2:,j] - I[1:-1:j])      \
                   -         (q[1:-1:j] + q[:-2:j]) * (I[1:-1:j]   - I[:-2:j])) \
                   + dtdy2*2*(q[1:-1:j] + q[1:-1:,j+1]) * (I[1:-1:,j+1] - I[1:-1:j]))

        j = Iy[-1] # 1) Boundary where y = Ny
            u[1:-1:j] = 0.5*(2*I[1:-1:j] - 2*B*dt*V[1:-1:j] + dt2*f[1:-1:j]     \
                   + dtdx2*( (q[1:-1:j] + q[2:,j]) * (I[2:,j] - I[1:-1:j])      \
                   -         (q[1:-1:j] + q[:-2:j]) * (I[1:-1:j]   - I[:-2:j])) \
                   + dtdy2*2*(q[1:-1:j] + q[1:-1:,j-1]) * (I[1:-1:,j-1] - I[1:-1:j]))

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
                 + dtdx2*2*(q[i,1:-1] + q[i+1,:1:-1]) * (u_1[i+1,:1:-1] - u_1[i,1:-1])  \
                 + dtdy2*( (q[i,1:-1] + q[i,2:])      * (u_1[i,2:]      - u_1[i,1:-1])  \
                 -         (q[i,1:-1] + q[i,:-2])     * (u_1[i,1:-1]    - u_1[i,:-2])))

          i = Ix[-1] # 1) Boundary where x = Nx
          u[i,1:-1] = A*( 2*u_1[i,1:-1] + B*u_2[i,1:-1] + dt2*f[i,1:-1]                 \
                 + dtdx2*2*(q[i,1:-1] + q[i-1,:1:-1]) * (u_1[i-1,:1:-1] - u_1[i,1:-1])  \
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
               -         (q[1:-1,j] + q[:-2,j])     * (u_1[1:-1,j]  - u_1[:-2,j]))  \
               + dtdy2*2*(q[1:-1,j] + q[1:-1,j-1]) * (u_1[1:-1,j-1] - u_1[1:-1,j]) )

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
    return u


def advance_scalar(u, u_1, u_2, q, f, x, y, t, n, A, B, dt2, dtdx2,dtdy2,
                   V=None, step1=False):
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

    return u


def quadratic(Nx, Ny, version):
    """Exact discrete solution of the scheme."""

    def exact_solution(x, y, t):
        return x*(Lx - x)*y*(Ly - y)*(1 + 0.5*t)

    def I(x, y):
        return exact_solution(x, y, 0)

    def V(x, y):
        return 0.5*exact_solution(x, y, 0)

    def f(x, y, t):
        return 2*c**2*(1 + 0.5*t)*(y*(Ly - y) + x*(Lx - x))

    Lx = 5;  Ly = 2
    c = 1.5
    dt = -1 # use longest possible steps
    T = 18

    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        tol = 1E-12
        msg = 'diff=%g, step %d, time=%g' % (diff, n, t[n])
        assert diff < tol, msg

    new_dt, cpu = solver(
        I, V, f, c, Lx, Ly, Nx, Ny, dt, T,
        user_action=assert_no_error, version=version)
    return new_dt, cpu


def test_quadratic():
    # Test a series of meshes where Nx > Ny and Nx < Ny
    versions = 'scalar', 'vectorized', 'cython', 'f77', 'c_cy', 'c_f2py'
    for Nx in range(2, 6, 2):
        for Ny in range(2, 6, 2):
            for version in versions:
                print 'testing', version, 'for %dx%d mesh' % (Nx, Ny)
                quadratic(Nx, Ny, version)

def run_efficiency(nrefinements=4):
    def I(x, y):
        return sin(pi*x/Lx)*sin(pi*y/Ly)

    Lx = 10;  Ly = 10
    c = 1.5
    T = 100
    versions = ['scalar', 'vectorized', 'cython', 'f77',
               'c_f2py', 'c_cy']
    print ' '*15, ''.join(['%-13s' % v for v in versions])
    for Nx in 15, 30, 60, 120:
        cpu = {}
        for version in versions:
            dt, cpu_ = solver(I, None, None, c, Lx, Ly, Nx, Nx,
                              -1, T, user_action=None,
                              version=version)
            cpu[version] = cpu_
        cpu_min = min(list(cpu.values()))
        if cpu_min < 1E-6:
            print 'Ignored %dx%d grid (too small execution time)' \
                  % (Nx, Nx)
        else:
            cpu = {version: cpu[version]/cpu_min for version in cpu}
            print '%-15s' % '%dx%d' % (Nx, Nx),
            print ''.join(['%13.1f' % cpu[version] for version in versions])

def gaussian(plot_method=2, version='vectorized', save_plot=True):
    """
    Initial Gaussian bell in the middle of the domain.
    plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
    """
    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    Lx = 10
    Ly = 10
    c = 1.0

    def I(x, y):
        """Gaussian peak at (Lx/2, Ly/2)."""
        return exp(-0.5*(x-Lx/2.0)**2 - 0.5*(y-Ly/2.0)**2)

    if plot_method == 3:
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.ion()
        fig = plt.figure()
        u_surf = None

    def plot_u(u, x, xv, y, yv, t, n):
        if t[n] == 0:
            time.sleep(2)
        if plot_method == 1:
            mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],
                 caxis=[-1,1])
        elif plot_method == 2:
            surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 1],
                  colorbar=True, colormap=hot(), caxis=[-1,1],
                  shading='flat')
        elif plot_method == 3:
            print 'Experimental 3D matplotlib...under development...'
            #plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            u_surf = ax.plot_surface(xv, yv, u, alpha=0.3)
            #ax.contourf(xv, yv, u, zdir='z', offset=-100, cmap=cm.coolwarm)
            #ax.set_zlim(-1, 1)
            # Remove old surface before drawing
            if u_surf is not None:
                ax.collections.remove(u_surf)
            plt.draw()
            time.sleep(1)
        if plot_method > 0:
            time.sleep(0) # pause between frames
            if save_plot:
                filename = 'tmp_%04d.png' % n
                savefig(filename)  # time consuming!

    Nx = 40; Ny = 40; T = 20
    dt, cpu = solver(I, None, None, c, Lx, Ly, Nx, Ny, -1, T,
                     user_action=plot_u, version=version)



if __name__ == '__main__':
    test_quadratic()
