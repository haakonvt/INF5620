from wave1D_dn_vc_modified import *
from numpy import cos,pi,log,sqrt,linspace
import sys
from matplotlib.pyplot import plot,show,figure

# Find f(x,t) with sympy:
def find_f(L_value):
    from sympy import symbols,cos,pi,diff,simplify,lambdify
    x,t,L = symbols('x t L')

    if task == 'a':
        q = lambda x: 1 + (x-L/2)**4
    elif task == 'b':
        q = lambda x: 1 + cos(pi*x/L)
    else:
        print '-------'
        print '`task` must be set to either a) or b) with syntax: task=`a` (or `b`)'
        sys.exit()

    u = lambda x,t: cos(pi*x/L)*cos(t)

    utt   = diff(u(x,t),t,t)
    du    = diff(u(x,t),x)
    dxqdu = diff(q(x)*du,x)

    fsym  = simplify(utt-dxqdu).subs(L,L_value)
    return lambdify((x, t), fsym, modules='numpy') # Return a as a non-symbolic function

def convergence_rate(u, x, t, n):
    L = 1.0
    u_e = cos(pi*x/L)*cos(t[n])
    #diff = max(u - u_e)
    e = u_e - u
    E = np.sqrt(dt*sum(e**2))
    E_list_t.append(E)

def animate(umin=-1,umax=1,skip_frame=1):
    action = PlotAndStoreSolution(umin=umin, umax=umax, screen_movie=True, skip_frame=skip_frame)
    return action

def I(x):
    return cos(pi*x/L)

def c(x):
    if task == 'a':
        return sqrt(1 + (x-L/2)**4)
    elif task == 'b':
        return sqrt(1 + cos(pi*x/L))
    else:
        print '-------'
        print '`task` must be set to either a) or b) with syntax: task=`a` (or `b`)'
        sys.exit()

#--------------------
global task; task = 'a' # Specify the task to be used (either a or b)
#--------------------
# Start the simulation with a single call
L           = 1.0 # Spatial domain
C           = 1.0 # Courant number
f           = find_f(L) # Using sympy and choose task a or b
Neumann_ver = False # Use standard Neumann bcs discretization (57) or False = (54)
Nx_values   = [100,200,500]
E_values    = [] # Will contain largest error for a specific run
dt_values   = []


user_action = convergence_rate
print "\nRunning convergence tests"
for Nxi in Nx_values: # Run all experiments
    E_list_t = []
    dx = float(L)/Nxi
    #c_max = max(c(linspace(0,L,Nxi))) # Both functions have maxima at c(x=0)
    dt =  C*dx/c(0)

    solver(I=I, V=None, f=f, c=c, U_0=None, U_L=None,
           L=L, dt=dt, C=C, T=3, user_action=user_action, version='vectorized',
           stability_safety_factor=1.0, use_std_neuman_bcs=Neumann_ver)

    dt_values.append(dt)
    E_values.append(max(E_list_t))

m = len(Nx_values)
r = [log(E_values[i-1]/E_values[i])/log(dt_values[i-1]/dt_values[i]) for i in range(1, m, 1)]
r = [round(r_, 4) for r_ in r] # Round to three decimals

# Print out convergence rates
print "-----------------------------------------------------------------"
print "Nx(i) | Nx(i+1) |  dt(i)    |  dt(i+1)  |   r(i) | Std. Neu. bcs?"
print "-----------------------------------------------------------------"
for i in range(m-1):
    print "%-3i     %-4i      %-9.3E   %-9.3E   %-5.4f   %s" \
        %(Nx_values[i], Nx_values[i+1], dt_values[i], dt_values[i+1], r[i], Neumann_ver)
figure(); plot(Nx_values[1:],r); show()

screen_animation = raw_input('\nAnimate a specific Nx-value, y/n? ')
if screen_animation == 'y':
    user_action = animate(umin=-1,umax=1,skip_frame=1)
    Nx = float(raw_input('Input a single value for Nx:'))
    dx = float(L)/Nx
    dt =  C*dx/c(0)
    solver(I=I, V=None, f=f, c=c, U_0=None, U_L=None,
           L=L, dt=dt, C=C, T=3, user_action=user_action, version='vectorized',
           stability_safety_factor=1.0, use_std_neuman_bcs=Neumann_ver)
