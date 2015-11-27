from dolfin import *
import numpy as np

def nld_solver(N,I,alpha,dt,T,rho,f,P):
    """
    Solves the non-linear diffusion problem, with Neuman bcs.:
    rho * u_t = div(a(u), grad(u)) + f

    Args:
      N    (list) : With mesh points as elements, i.e. for 2D: [50,50]
      I    (array): Initial condition at t=0
      alpha (func): The diffusion coefficient, may depend on u, i.e. non-linear
      dt   (float): The temporal resolution, aka the time step
      T    (float): Ending time of computation
      rho  (float): Constant in the equation related to the rate of diffusion
      f    (func) : The source term
      P    (int)  : Degree of finite elements

    Returns:
      Solution at final time step, T
    """
    # Only show output that could lead to, or be a problem:
    set_log_level(WARNING)

    # Define Neumann boundary conditions du/dn=0
    # Used by standard

    # Create mesh (1D,2D,3D)
    dim  = np.array(N).size
    mesh = UnitIntervalMesh(N[0]) if dim == 1 else UnitSquareMesh(N[0],N[1]) if dim == 2 \
           else UnitCubeMesh(N[0],N[1],N[2])

    # Create function space V, using 'continous Galerkin'..
    # ..implying the standard Lagrange family of elements
    # P1 = 1, P2 = 2 etc.
    V = FunctionSpace(mesh, 'CG', P)

    # Use initial condition to specify previous (first) known u
    u_1 = interpolate(I, V)

    # Define variables for the equation on variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (u*v + dt/rho*inner(alpha(u_1)*nabla_grad(u), nabla_grad(v)))*dx
    L = (u_1 + dt/rho*f)*v*dx

    # Run simulation:
    t = 0; t += dt
    u = Function(V) # The new solution
    while t < (T + 1E-8):
        f.t = t # Update time to current in source term
        # Do a single Picard iteration
        solve(a == L, u)
        u_1.assign(u) # Update u_1 for next iteration
        t += dt
    return u_1,V


def task_d():
    """
    Testing a constant solution
    """
    N = 5 # 2D mesh. can also be 1D and 3D
    I = Expression('1.0')# some initial condition
    alpha = lambda u: 2
    f = Constant('0')
    dt = 0.1; T=1; rho = 1
    interval,square,box = [5], [5,5], [5,5,5]
    for P in [1,2]:
        for N in interval,square,box:
            u,V = nld_solver(N,I,alpha,dt,T,rho,f,P)
            u_e = interpolate(I, V)
            abs_err = np.abs(u_e.vector().array() - u.vector().array()).max()
            print 'Using P%d in %dD, err.: %e' % (P,np.array(N).size, abs_err)



if __name__ == '__main__':
    task_d()

    """
    wiz = plot(u, interactive = True)
    wiz.write_png('t_%d_%d' %(P,np.array(N).size))
    """
