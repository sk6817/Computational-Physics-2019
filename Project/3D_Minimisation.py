'''
Computational Physics Project
A log-likelihood fit for extracting neutrino oscillation parameters

Part 4: 3D Minimisation
'''
# importing variables and functions from previous .py file
from The_Data import events, sim, p, x, np, plt, lamb

# defining new lambda considering neutrino cross section

def lamb_2(u):
    
    '''Returns the expected event rate in each bin using the parameters, u.
    
    Parameters
    ---------------
        u (list): A list of parameters; theta, m_squared and gamma
        
    Returns
    --------------
        A list containing 200 values (float) of expected event rate.   
    
    '''
        
    L = 295
    theta = u[0]           # first element is theta
    m_sq = u[1]            # second element is m_squared
    gamma = u[2]           # third element is gamma (new variable)
    
    lamb_list = []
    for i in range(200):           # modified to include gamma
        lamb = sim[i] * p(theta, m_sq, L, x[i]) * gamma * x[i]
        lamb_list.append(lamb)
    
    return lamb_list

# defining new likelihood in 3D
    
def likelihood_3d(u):
    
    '''Returns the Negative Log likelihood of the events for a given set of three
        parameters: theta, m_squared and gamma.
       
    Parameters
    ----------------
        u (list) : A list of three parameters (theta, m_squared and gamma)
        
    Returns
    ----------------
        The Negative Log likelihood of the three variable variables (float)   
    '''
    
    nll = []
    k = events 
    
    for i in range(200):
         
        if k[i] > 0:                        # similarly to avoid log(0)
            lamb = lamb_2(u)[i]             # modified lambda function
            nll_i = lamb - k[i] + (k[i] * np.log( k[i] / lamb))
            nll.append(nll_i)
            
        if k[i] == 0:
            lamb_0 = lamb_2(u)[i]
            nll.append(lamb_0)
             
    return sum(nll)

# plotting nll_3d against gamma --> using guesses for theta and m_squared
    
gamma_range = np.linspace(0.1, 4,100)    #  raneg for gamma for plotting

ax13 = plt.figure(13).add_subplot(111)
ax13.set(title='NLL against $\gamma$ with a fixed $\Theta_{23}$ and $\Delta {m^2}_{23}$',
           xlabel= r"$\gamma$ ($GeV^{-1}$)",
           ylabel='NLL') 

plt.plot(gamma_range, likelihood_3d([0.67, 0.00259, gamma_range]), color = 'red')
plt.show()

#%%    
# 3D minimisation using gradient descent
def grad_3d(f, x, y, z):
    
    '''Returns the gradient vector of a function, f at a given point (x,y,z).
       
    Parameters
    -----------------------        
        f : A function of 3 variables
        x (float) : The x of the point  (theta)
        y (float) : The y of the point  (m_sq)
        z (float) : The z of the point  (gamma)

    Returns
    -----------------------
        The gradient vector; each element is for each direction (3 directions).
    '''
    
    delta_x = 1e-5                            # the step size in x
    df_dx = ( f([ x + delta_x, y, z]) - f([x, y, z]) ) / delta_x
    
    delta_y = 1e-9
    df_dy = ( f([ x, y + delta_y, z]) - f([x, y, z]) ) / delta_y
    
    delta_z = 1e-5
    df_dz = ( f([ x, y, z + delta_z]) - f([x, y, z]) ) / delta_z
    
    return np.array([df_dx, df_dy, df_dz])

def grad_minimiser_3d(f, guess, alpha, iterations):
    
    '''Returns the minimum of a function, f(x,y,z).
       
    Parameters
    -----------------------        
        f : A function of 3 variables, f(x,y,z).
        guess (list) : A list of two elements; one for x guess, one for y guess.
        alpha (array) : The step size in three directions; x, y and z
        iterations : Number of steps in gradient descent

    Returns
    -----------------------
        (x_min, y_min, z_min) : The minimum point of f(x,y,z)
        loss_list (list) : A list of f(x_min, y_min, z_min) at each step
    '''    
    a = alpha                  # step size in gradient descent

    max_it = iterations        # max num of iterations
    it = 0                     # iterations
    x_0 = guess
    loss_list = []
    
    while it < max_it :
        
        # similar to 2D but f is modified to take 3 variables in
        x_0 = x_0 - a.T * grad_3d(f, x_0[0], x_0[1], x_0[2])
        
        it = it + 1
        loss_list.append(f(x_0))
        
        print(it, x_0, f(x_0))
           
    return x_0, loss_list   

#%%
    
# running grad descent on likelihood_3d    (takes about 70 seconds to run)
alpha3 = np.array([1e-4, 1e-9, 1e-3]) 
g_vector_3d_1 = [0.6, 2.4e-3, 0.6]
min_3d_1, loss_list_1 = grad_minimiser_3d(likelihood_3d, g_vector_3d_1, alpha3, 30) 
                   
g_vector_3d_2 = [0.9, 2.4e-3, 0.6]
min_3d_2, loss_list_2 = grad_minimiser_3d(likelihood_3d, g_vector_3d_2, alpha3, 30)   

#plotting steps 

ax14 = plt.figure(14).add_subplot(111)
ax14.set(title='NLL after each step',
           xlabel= 'Number of step',
           ylabel='NLL')
    
step = np.arange(0, len(loss_list_1))    
plt.plot(step, loss_list_1, 'r', label = 'First minimum') 
plt.plot(step, loss_list_2, 'b', label = 'Second minimum') 
plt.legend()
plt.show()  

print('The first global min is', min_3d_1)    
print('The second global min is', min_3d_2)                 

#%%
# importing scipy to check
import scipy.optimize as op
op.minimize(likelihood_3d, [0.6, 2.4e-3, 0.6], method='nelder-mead')
#op.minimize(likelihood_3d, [0.9, 2.4e-3, 0.6], method='nelder-mead')

# scipy results  ----> checks out
# x1: [0.67508737, 0.00272718, 1.60060254]
# x2: [0.89572927, 0.00272722, 1.60063676]

#%%
#uncertainty

def error_3d(f, x_min, axis):
    
    '''Returns the error of a function f(x,y,z) at a given minimum point,
      (x_min, y_min, z_min) for a given axis.
    
    Parameters
    -------------------
        f : A function of three variables f. Here f = likelihood_3d
        min_point (list) : The minimum point [x_min, y_min, z_min] of f
        axis (str) : The direction in which the error to be determined
        
    Returns
    ----------------
        std (float) : The standard deviation or the error    
    '''
    #  similarly to 2D case
    
    theta_min = x_min[0]
    m_min = x_min[1]
    gamma_min = x_min[2]
    l_min = f(x_min)
    l_plus = l_min + 0.5
    
    if axis == 'theta':
        d = 0.1
        criteria = 0.1
        x_range = np.linspace(theta_min - d, theta_min + d, 200)
        err_list = []
        
        for i in x_range:
            if (abs(f([i, m_min, gamma_min]) - l_plus) < criteria):
                err_list.append(theta_min - i)
        err_plus = max(err_list)
        err_minus = abs(min(err_list))  
        std = (err_minus + err_plus) /2

    if axis == 'm':
        criteria = 0.1
        m_range = np.linspace(0.002, 0.003, 200)
        err_list = []
        
        for i in m_range:
            if (abs(f([theta_min, i, gamma_min]) - l_plus) < criteria):
                err_list.append(m_min - i)
        
        err_plus = max(err_list)
        err_minus = abs(min(err_list)) 
        std = (err_minus + err_plus) /2
   
    if axis == 'gamma':
        criteria = 0.1
        gamma_range_e = np.linspace(1.50, 1.70, 200)
        err_list_g = []
        
        for i in gamma_range_e:
            if (abs(likelihood_3d([theta_min, m_min, i]) - l_plus) < 0.01):
                err_list_g.append(min_3d_1[2] - i) 
                
        err_plus = max(err_list_g)
        err_minus = abs(min(err_list_g)) 
        std = (err_minus + err_plus) /2
         
    return std

#%% 
# Calculating the error and quoting the full result
# first minimum    (takes about 90 seconds to run)
    
err_3d_1_theta = error_3d(likelihood_3d, min_3d_1, 'theta')
print('The first theta minimum is', round(min_3d_1[0], 4), '+-', round(err_3d_1_theta, 4))

err_3d_1_m = error_3d(likelihood_3d, min_3d_1, 'm')    
print('The first m minimum is', round(min_3d_1[1], 6), '+-', round(err_3d_1_m, 6))

err_3d_1_gamma = error_3d(likelihood_3d, min_3d_1, 'gamma')
print('The first gamma minimum is', round(min_3d_1[2], 4), '+-', round(err_3d_1_gamma, 4))

#%%
# second minimum     (takes about 90 seconds to run)

err_3d_2_theta = error_3d(likelihood_3d, min_3d_2, 'theta')
print('The second theta minimum is', round(min_3d_2[0], 4), '+-', round(err_3d_2_theta, 4))

err_3d_2_m = error_3d(likelihood_3d, min_3d_2, 'm')    
print('The second m minimum is', round(min_3d_2[1], 6), '+-', round(err_3d_2_m, 6))        

err_3d_2_gamma = error_3d(likelihood_3d, min_3d_2, 'gamma')
print('The second gamma minimum is', round(min_3d_2[2], 4), '+-', round(err_3d_2_gamma, 4))

#%%
# plotting the fit curve for all results

parameters = [0.675, 2.727e-3, 1.600]    #the other min is identical

E_step = np.linspace(0, 10, 200)
ax15 = plt.figure(15).add_subplot(111)
ax15.set(title= 'Event Distribution',
           xlabel='Energy (GeV)',
           ylabel='Event Rate',
           xlim = (0,10)) 

#plt.plot(x, sim, color = 'yellow', label = 'no oscillation')
plt.step(E_step, events, color = 'red', label = 'Observation')
plt.step(E_step, lamb([0.6626355,2.4e-3]), color = 'orange', label = '1D Fit')
plt.step(E_step, lamb([0.67032787,0.00259088]), color = 'green', label = '2D Fit')
plt.step(E_step, lamb_2(parameters), color = 'blue', label = '3D Fit')
plt.legend()
plt.show()