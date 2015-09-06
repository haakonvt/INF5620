import numpy as np
import matplotlib.pyplot as plt

def simulate(
    beta=0.9,                 # dimensionless parameter
    Theta=30,                 # initial angle in degrees
    epsilon=0,                # initial stretch of wire
    num_periods=6,            # simulate for num_periods
    time_steps_per_period=60, # time step resolution
    plot=True,                # make plots or not
    ):

	b     = beta
	theta = np.deg2rad(Theta)
	Nt    = num_periods * time_steps_per_period
	dt    = 2 * np.pi   / time_steps_per_period	
	x     = np.zeros(Nt+1)
	y     = np.zeros(Nt+1)
	t     = np.zeros(Nt+1)

	# Initial conditions
	x[0]  = (1+epsilon)*np.sin(theta)
	y[0]  = 1-(1+epsilon)*np.cos(theta)

	L     = np.sqrt( x[0]**2 + (y[0]-1)**2 )
	c     = b/(1-b)   # For better readability

	# First time step
	x[1]  = 0.5*(2*x[0] - dt**2*x[0]*c*(1-b/L))
	y[1]  = 0.5*(2*y[0] - dt**2*(b + (y[0]-1)*c*(1-b/L)))
	t[1]  = dt
	
	# Loop over all future steps
	for i in range(1,Nt):
		L      = np.sqrt( x[i]**2 + (y[i]-1)**2 )
		x[i+1] = 2*x[i] - x[i-1] - dt**2*( x[i]   *c*(1-b/L)    )
		y[i+1] = 2*y[i] - y[i-1] - dt**2*((y[i]-1)*c*(1-b/L) + b)

		t[i+1] = (i+1)*dt
	
	thetas =  np.arctan2(x, 1-y)
	
	if plot == True:
		
		plt.figure(1)
		plt.subplot(211)
		plt.plot(x,y)
		plt.gca().set_aspect('equal') # Set axis equal
		plt.legend(["Motion of pendulum [x,y]"])
		
		plt.subplot(212)
		plt.plot(t,thetas)
		plt.legend(["Theta(t)"])

		if Theta < 10:
			plt.figure(2)
			beta_nel = 0.995 # Non-elastic pendulum
			x_nel,y_nel,theta_nel,t = simulate(
    						beta_nel,  
					        Theta,
    						epsilon,
    						num_periods,
    						time_steps_per_period,
    						plot=False)

			# Compare elastic vs non-elastic pendulum
			plt.subplot(211)
			plt.gca().set_aspect('equal') # Set axis equal
			plt.plot(x,y,'b',x_nel,y_nel,'r')
			plt.legend(["Elastic pendulum","Non-elastic pendulum"])
			plt.title("Comparison with non-elastic pend.")
			plt.subplot(212)
			plt.plot(t,thetas,'b',t,theta_nel,'r')
			plt.legend(["Elastic pendulum","Non-elastic pendulum"])

		plt.show()
	return x, y, thetas, t 


def test_function_1():
	x,y,theta,t = simulate(
    				beta=0.8, 
				Theta=0.0,
    				epsilon=0.0,
    				num_periods=6,
    				time_steps_per_period=80,
    				plot=False)
	xsum = sum(abs(x)) # By first taking |x|, periodic effects wont cancel out
	ysum = sum(abs(y))
	tol  = 1e-15
	if xsum < tol and ysum < tol:
		print "Test with theta = eps = 0 is passed with tolerance %s" %tol
		print "Sum(x) = %s, sum(y) = %s" %(xsum,ysum)
		return True
	else:
		return False


def test_function_2(time_steps_per_period = 500, tol=1e-5, visualize_error=False):
	beta  = 0.8
	theta = 0.0
	epsilon = 0.2
	num_periods = 2
	x,y,theta,t = simulate(beta, theta, epsilon,num_periods,time_steps_per_period,plot=False)

	# Exact solution
	w   = np.sqrt(beta/(1-beta))
	y_e = -epsilon*np.cos(w*t)

	abs_diff = abs(y-y_e)
	error    = max(abs_diff)
	
	if error < tol:
		print "Test passed. Max. error: %.3E < tolerance: %s" %(error,tol)
		if visualize_error == True:
			plt.figure(3)
			plt.subplot(211)
			plt.plot(t,y,'b',t,y_e,'r')
			plt.legend(["Numerical solution, y_n(t)","Exact solution, y_e(t)"])
			plt.title("Test of restricted motion [y-direction]")
	
			plt.subplot(212)
			plt.plot(t,abs_diff,'k')
			plt.legend(["Absolute error, abs(y-y_e)"])
		
			plt.show()
		return True
	else:
		print "Test failed. Max. error %.3E > tolerance, %s" %(error,tol)
		return False


def demo(beta,theta):
	epsilon = 0.0
	num_periods = 3
	time_steps_per_period = 600
	x,y,theta,t = simulate(beta, theta, epsilon, num_periods, time_steps_per_period)


if __name__ == '__main__':
	print "-------------------------"
	print "Doing unit tests 1 / 2..."
	# Test that theta = eps = 0 gives x = y = 0 for all t
	test1 = test_function_1() 
	print "-------------------------"
	print "Doing unit tests 2 / 2..."
	# Test that pure y-motion gives correct results
	test2 = test_function_2(5000,visualize_error=True)
	print "-------------------------"
	if test1 == True and test2 == True:
		print "All unit test were successful!"
		print "-------------------------"

		# Test-run with specific paramters
		beta, theta  = 0.8, 9 
		demo(beta, theta)











