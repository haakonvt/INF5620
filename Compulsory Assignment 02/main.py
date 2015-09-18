from wave1D_dn_vc import *

# Find f(x,t) with sympy:
def find_f(L_value):
    from sympy import symbols,cos,pi,diff,simplify,lambdify
    x,t,L = symbols('x t L')

    q = lambda x   : 1 + (x-L/2)**4
    u = lambda x,t: cos(pi*x/L)*cos(t)

    utt   = diff(u(x,t),t,t)
    du    = diff(u(x,t),x)
    dxqdu = diff(q(x)*du,x)

    fsym  = simplify(utt-dxqdu).subs(L,L_value)

    return lambdify((x, t), fsym, modules='numpy')


def wave_solver(C=1, Nx=200, animate=True, version='vectorized', T=2,
          loc='center', pulse_tp='gaussian', slowness_factor=2,
          medium=[0.7, 0.9], every_frame=1, sigma=0.05):

    L = 1.0

    if pulse_tp == 'test_exact':
        def I(x):
            return cos(pi*x/L) # TASK 13 a)
            #return sin(3*pi*x/L) # Use Dirichlet bcs

    def c(x):
        return 1

    f = find_f(L)

    umin=-1; umax=1
    casename = '%s_Nx%s_sf%s' % \
               (pulse_tp, Nx, slowness_factor)
    action = PlotMediumAndSolution(
        medium, casename=casename, umin=umin, umax=umax,
        every_frame=every_frame, screen_movie=animate)

    solver(I=I, V=None, f=f, c=c, U_0=None, U_L=None,
           L=L, Nx=Nx, C=C, T=T,
           user_action=action, version=version,
           dt_safety_factor=1)

# Start the simulation with a single call
wave_solver(T=3, pulse_tp='test_exact', Nx=100, medium=[0, 1],every_frame=5, slowness_factor=1)
