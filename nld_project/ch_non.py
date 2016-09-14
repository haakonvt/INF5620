from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def solver(I, dt, f, rho, alpha, degree, divisions, T=1, u_exact=None, user_action=None):
    """Solver for the non-linear diffusion equation."""

    set_log_level(WARNING)

    #Define mesh:
    d = len(divisions)
    domain = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    mesh = domain[d-1](*divisions)

    V = FunctionSpace(mesh, 'CG', degree)

    u = TrialFunction(V)
    v = TestFunction(V)
    u1 = interpolate(I, V)

    u_k = u1

    #Variational form:
    a = u*v*dx + dt/rho*inner(alpha(u_k)*nabla_grad(u), nabla_grad(v))*dx
    L = (u1 + dt/rho*f)*v*dx

    u = Function(V)
    t = dt
    Err = []
    while t <= T:

        f.t = t
        if u_exact:
            u_exact.t = t
            u_e = interpolate(u_exact, V)
        else:
            u_e = None

        #Picard iterations:
        eps = 1.0
        tol = 0
        iterations = 0
        maxiter = 1

        while eps>tol and iterations<maxiter:
            iterations += 1
            solve(a == L, u)
            diff = u.vector().array() - u_k.vector().array()
            eps = np.linalg.norm(diff, ord=np.Inf)
            u_k.assign(u)

        #User action is different for each task:
        if user_action:
            E = user_action(u, u_e, t, dt)
            Err.append(E)

        #Update previous step:
        u1.assign(u)
        t += dt
    return Err[-1]#max(Err)

def test_exercise_d():
    """Verification with constant solution."""

    u_exact = Constant("3.0")
    I = Constant("3.0")
    dt = 0.5
    f = Constant("0")
    rho = 10
    alpha = lambda u: 5
    N = 5
    degrees = [1, 2]
    dimensions = [1, 2, 3]

    def error_const(u, u_e, t, dt):
        error = abs(u_e.vector().array() - u.vector().array()).max()
        assert error < 10**(-12)
        print "t=%.2f, error=%s" % (t, error)

    for degree in degrees:
        for dims in dimensions:
            print "\nP%i elements;" %degree, "%i dimensions" % dims
            divisions = [N]*dims
            solver(I, dt, f, rho, alpha, degree, divisions,
                   u_exact=u_exact, user_action=error_const)


def test_exercise_e():
    """Verification with simple analytical solution."""

    u_exact = Expression("exp(-pi*pi*t)*cos(pi*x[0])", t=0)
    I = Expression("cos(pi*x[0])")
    f = Constant("0")
    rho = 1
    alpha = lambda u: 1
    degree = 1
    dimensions = 2
    T = 0.1
    dt = 0.1
    print "\nP%i elements;" %degree, "%i dimensions" % dimensions

    #Calculate error:
    def return_error(u, u_e, t, dt):
        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size)
        return E

    #Calculate E/h for varying h:
    for i in range(0, 7):
        N = int(round(1./sqrt(dt)))
        divisions = [N]*dimensions
        E = solver(I, dt, f, rho, alpha, degree, divisions,
                  T=T, u_exact=u_exact, user_action=return_error)
        h = dt
        print "h=%f, E/h=%f, N=%i" % (h, E/h, N)
        dt /= 2.

def test_exercise_f():
    """Verification with source term."""

    u_exact = Expression("t*x[0]*x[0]*(0.5 - x[0]/3.)", t=0)
    I = Constant("0")
    rho = 1
    f = Expression("""-rho*x[0]*x[0]*x[0]/3 + rho*x[0]*x[0]/2
                      + 8*t*t*t*pow(x[0], 7)/9
                      - 28*t*t*t*pow(x[0], 6)/9
                      + 7*t*t*t*pow(x[0], 5)/2
                      - 5*t*t*t*pow(x[0], 4)/4
                      + 2*t*x[0] - t""", t=0, rho=rho)
    alpha = lambda u: 1 + u**2
    T=1.5
    dt = 0.5
    N = 20
    degree = 1
    dimensions = 1
    divisions = [N]*dimensions

    #Plot for comparison:
    def plot_comparison(u, u_e, t, dt):
        x = np.linspace(0,1,u.vector().array().size)
        plt.plot(x, u_e.vector().array()[::-1], '-')
        plt.plot(x, u.vector().array()[::-1], 'o')
        plt.title("t=%s" %t)
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend(["Exact solution", "Numerical solution"], loc="best")
        plt.show()

    solver(I, dt, f, rho, alpha, degree, divisions,
           T=T, u_exact=u_exact, user_action=plot_comparison)

def test_exercise_h():
    """Verification by checking convergance rates."""

    u_exact = Expression("t*x[0]*x[0]*(0.5 - x[0]/3.)", t=0)
    I = Constant("0")
    dt = 0.5
    rho = 1
    f = Expression("""rho*x[0]*x[0]*(-2*x[0] + 3)/6
                     -(-12*t*x[0] + 3*t*(-2*x[0] + 3))
                     *(pow(x[0], 4)*(-dt + t)*(-dt + t)
                     *(-2*x[0] + 3)*(-2*x[0] + 3) + 36)/324
                     -(-6*t*x[0]*x[0] + 6*t*x[0]
                     *(-2*x[0] + 3))*(36*pow(x[0], 4)
                     *(-dt + t)*(-dt + t)*(2*x[0] - 3)
                     +36*x[0]*x[0]*x[0]*(-dt + t)
                     *(-dt + t)*(-2*x[0] + 3)
                     *(-2*x[0] + 3))/5832""",
                     t=0, dt=dt, rho=rho)
    alpha = lambda u: 1 + u**2
    degree = 1
    dimensions = 1

    #Calculate error:
    def return_error(u, u_e, t, dt):
        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size).max()
        return E

    errors = []
    dt_values = []

    for i in range(0, 10):
        N = int(round(1./sqrt(dt)))
        divisions = [N]*dimensions

        E = solver(I, dt, f, rho, alpha, degree, divisions, u_exact=u_exact, user_action=return_error)
        dt_values.append(dt)
        errors.append(E)
        dt /= 2.
        f.dt = dt

    #Calculate convergance rates:
    def compute_rates(dt_values, errors):
        m = len(errors)
        #Convergence rate:
        r = [np.log(errors[i-1]/errors[i])/
             np.log(dt_values[i-1]/dt_values[i])
             for i in range(1, len(errors))]

        return r

    conv_rates = compute_rates(dt_values, errors)

    print "\nConvergance rates:"
    for i in range(len(conv_rates)):
        print "h1=%f, h2=%f,   r=%f" % (dt_values[i], dt_values[i+1], conv_rates[i])

def test_exercise_i():
    """Simulate nonlinear diffusion of Gaussian function."""

    sigma = .5
    I = Expression("exp(-1./(2*sigma*sigma)\
                        *(x[0]*x[0] + x[1]*x[1]))",
                   sigma=sigma)
    T = 0.2
    dt = 0.002
    f = Constant("0")
    rho = 1.
    beta = 10.
    alpha = lambda u: 1 + beta*u**2
    N = 40
    degree = 1
    dimensions = 2
    divisions = [N]*dimensions

    #Animate the diffusion of the surface:
    def plot_surface(u, u_e, t, dt):
        from time import sleep
        sleep(0.05)
        fig = plot(u)#.set_min_max(0,0.83)
        fig.set_min_max(0,0.83)
        #Save initial state and equilibrium state:
        if t==dt or t>T-dt:
            fig.write_png("plots/t%s" %t)

    solver(I, dt, f, rho, alpha, degree, divisions, T=T, user_action=plot_surface)


if __name__ == "__main__":
    #test_exercise_d()
    #test_exercise_e()
    #test_exercise_f()
    test_exercise_h()
    #test_exercise_i()
