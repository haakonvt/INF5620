from wave1D_dn_vc_modified import *
from numpy import cos,pi,log

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


def wave_solver(user_action, use_std_neuman_bcs, C=1.0, Nx=200, animate=True, version='vectorized', T=2):

    def I(x):
        return cos(pi*x/L)

    def c(x):
        return np.sqrt(1 + (x-L/2.)**4)

    L  = 1.0 # Spatial domain
    dx = float(L)/Nx
    dt =  C*dx/c(0)

    f = find_f(L) # Using sympy

    umin=-1; umax=1

    solver(I=I, V=None, f=f, c=c, U_0=None, U_L=None,
           L=L, dt=dt, C=C, T=T, user_action=convergence_rate, version='vectorized',
           stability_safety_factor=1.0,use_std_neuman_bcs=use_std_neuman_bcs)


def convergence_rate(u, x, t, n):
    L = 1.0
    u_e = cos(pi*x/L)*cos(t[n])
    diff = max(u - u_e)
    e = u_e - u
    dt = t[1]-t[0]
    if n==0:
        dt_values.append(dt)
    E = np.sqrt(dt*sum(e**2))
    E_list_t.append(E)

# Start the simulation with a single call
Neumann_ver = True # Use standard Neumann bcs discretization
Nx_values   = [10,20,30,60,90,120,150,180,500,1000]
E_values    = [] # Will contain largest error for a specific run
dt_values   = []
for Nxi in Nx_values: # Run all experiments
    E_list_t = []
    wave_solver(user_action=convergence_rate, use_std_neuman_bcs=Neumann_ver, T=3, Nx=Nxi)
    E_values.append(max(E_list_t))

m = len(Nx_values)
r = [log(E_values[i-1]/E_values[i])/log(dt_values[i-1]/dt_values[i]) for i in range(1, m, 1)]
r = [round(r_, 4) for r_ in r] # Round to three decimals

# Print out convergence rates
print "-------------------------------------------------------------------"
print "Nx(i) | Nx(i+1) | dt(i)     | dt(i+1)   |  r(i)  | Other Neu. bcs?"
print "-------------------------------------------------------------------"
for i in range(m-1):
    print "%-3i     %-4i      %-9.3E   %-9.3E   %-5.4f   %s" \
        %(Nx_values[i], Nx_values[i+1], dt_values[i], dt_values[i+1], r[i], Neumann_ver)
