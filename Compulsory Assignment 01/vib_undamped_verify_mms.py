import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sys
V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = 0.5*(2*(u(t+dt) - u(t))/dt + dt*w**2*u(t) - (2*V + dt*f))
    R = R.subs(t, 0) # Substitute t with 0
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt) - 2*u(t) + u(t-dt))/(dt*dt) 

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '=== Testing exact solution: %s ===' % u(t)
    print "Initial conditions u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))
    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)

def linear():
    main(lambda t: V*t + I)
    sys.exit("Run quad() in stead of linear if you want numerical solution and plot!")

def quad(a,b,c):
	global u
	u = lambda t: a*t**2 + b*t + c
	main(u)

def deg3():
	main(lambda t: t**3 + t**2 + t + 1)
	sys.exit("Run quad() in stead of deg3() if you want numerical solution and plot!")

def solver(dt_n,T,u0,dudt0,w_n,N):
	u_n = np.zeros(N) 
	# Use initial conditions
	u_n[0] = u0
	f0     = float(f.subs([(w, w_n), (t, 0)]))
	u_n[1] = 0.5*(2*dudt0*dt_n + f0*dt_n**2 +u_n[0]*(2-w_n**2*dt_n**2))

	for i in range(1,N-1):
		f_i      = float(f.subs([(w, w_n), (t, i*dt_n)]))
		u_n[i+1] = 2*u_n[i] - u_n[i-1] + dt_n**2*(-w_n**2*u_n[i] + f_i)

	return u_n

def nose_test(u_n, u_e, tol):
	diff_avg = sum(abs(u_n-u_e))/len(u_n)
	if diff_avg < tol:
		print "Nose test passed. Avg. abs. error: %.3E . Tolerance set to: %s " %(diff_avg,tol)
	else:
		print "Nose test failed. Avg. abs. error: %.3E . Tolerance set to: %s " %(diff_avg,tol)

if __name__ == '__main__':	
	# Solve problem numerically
	dt_n  = 0.1 # Time step
	T     = 2    # How long to simulate
	u0    = 1   # Init. cond.s.
	dudt0 = 1
	w_n   = 1
	N   = int(round(float(T/dt_n)))

	# --- Compute residuals ---
	#linear()
	quad(a=1,b=dudt0,c=u0)
	#deg3()
	
	t_vals    = np.linspace(0,T-dt_n,int(N)) 
	u_n = solver(dt_n,T,u0,dudt0,w_n,N)
	u_e   = u(t_vals)

	# Test the numerical solution to be exact
	tol = 1e-14
	nose_test(u_n,u_e,tol)

	plt.subplot(211)
	plt.plot(t_vals, u_n, '-b', t_vals, u_e, 'ro')
	plt.legend(['Numerical solution','Exact solution'])
	plt.subplot(212)
	plt.plot(t_vals, abs(u_n-u_e))
	plt.show()












