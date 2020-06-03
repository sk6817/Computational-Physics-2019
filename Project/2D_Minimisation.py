'''
Computational Physics Project
A log-likelihood fit for extracting neutrino oscillation parameters

Part 3: 2D Minimisation
'''

# importing variables and functions from previous The_Data.py file
from The_Data import events, sim, p, x, np, plt

# redefining likelihood function with two variables
def likelihood_2d(u):
    
    '''Returns the Negative Log likelihood of the events for a given set of two
        parameters: Theta and m_squared.
       
    Parameters
    ----------------
        u (list) : A list of two parameters (theta and m_squared)
        
    Returns
    ----------------
        The Negative Log likelihood of the two variables (float)   
    
    '''
    L = 295                     # fixed
    theta = u[0]                # first elememnt of u is theta
    m_sq = u[1]                 # second elememnt of u is m_squared
    nll = []
    k = events 
    
    for i in range(200):               # similar to likelihood in 1D
        
        if k[i] > 0:
            lamb = sim[i] * p(theta, m_sq, L, x[i])
            nll_i = lamb - k[i] + (k[i] * np.log( k[i] / lamb))
            nll.append(nll_i)
            
        if k[i] == 0:
            lamb_0 = sim[i] * p(theta, m_sq, L, x[i])
            nll.append(lamb_0)
    return sum(nll)


#plotting a surface plot ---> for visualisation purpose, not for report
    
from mpl_toolkits import mplot3d

thet_range = np.linspace(0, np.pi/2,200)        # range for theta
m_sq_range = np.linspace(0, 1e-2,200)           # range for m_squared
thet_range, m_sq_range = np.meshgrid(thet_range, m_sq_range)   # meshgrid of x, y

fig = plt.figure(8)
ax8 = plt.gca(projection='3d')
z_map = likelihood_2d([thet_range, m_sq_range])    # the likelihood in 2D

ax8.plot_surface(m_sq_range,thet_range, z_map,  cmap='jet')

ax8.set(title='NLL against $\Delta {m^2}_{23}$ and $\Theta_{23}$',
           xlabel= r"$\Delta {m^2}_{23}$ ($eV^2$)",
           ylabel= r'$\Theta_{23}$ (rad)',
           zlabel= 'NLL') 
plt.show()


#2d contour plot ---> for the report

fig_c, ax_c = plt.subplots()
m_sq_range_2 = [x*1e3 for x in m_sq_range]    # resale for plotting
cs = ax_c.contourf(thet_range, m_sq_range_2, z_map, 300,  cmap = 'jet')
cbar = fig_c.colorbar(cs)
ax_c.set(title='NLL against $\Delta {m^2}_{23}$ and $\Theta_{23}$',
           xlabel= r'$\theta_{23}$ (rad)',
           ylabel=  r"$\Delta {m^2}_{23}$ ($ \times 10^{-3} eV^2$)") 
ax_c.grid(False)
plt.show()


# plotting nll against m_sq alone, to see more clearly --> not really necessary

ax20 = plt.figure(20).add_subplot(111)
ax20.set(title='NLL against $\Delta {m^2}_{23}$',
           xlabel= r"$\Delta {m^2}_{23}$ ($ \times 10^{-3}eV^2$)",
           ylabel= 'NLL')
plt.plot(m_sq_range_2, likelihood_2d([0.67,m_sq_range]), 'b')

plt.show()

#%%

# defining minimiser in x and y direction (for simplicity)

def x_min_x(f, z, s):
    
    '''Returns the minimum point of a parabolic function that contains the three
       points (x, f(x,y)) in the list of z in x direction.
       
    Parameters
    --------------------------
        f : A function of 2 variable that returns f (float) at the three x's
        z (list) : A list of three x values (float)
        s (float) : The other variable to be inputted in f
    
    Returns
    --------------------------
        x_min_x (float) : The x that gives the minimum of a parabolic function in x direction
        
    NOTE: f is not necessarily parabolic, but is nearly parabolic at its minimum
    '''
    # as in the 1D case but the f has two variables
    
    x0 = z[0]
    y0 = f([x0, s])    
    x1 = z[1]
    y1 = f([x1, s])
    x2 = z[2]
    y2 = f([x2, s])
    
     # using Langrage polynomials
     
    x_min_x = (1/2) * ((x2**2 - x1**2) *y0 + (x0**2 - x2**2) *y1 + (x1**2 - x0**2) *y2) / ((x2 - x1) *y0 + (x0 - x2) *y1 + (x1 - x0) *y2)
    
    return x_min_x

# similarly for y direction    
def x_min_y(f, z, s):
    
    '''Returns the minimum point of a parabolic function that contains the three
       points (y, f(x, y)) in the list of z in y direction.
       
    Parameters
    --------------------------
        f : A function of 2 variable that returns f (float) at the three x's
        z (list) : A list of three y values (float)
        s (float) : The other variable to be inputted in f
    
    Returns
    --------------------------
        x_min_y (float) : The y that gives the minimum of a parabolic function in y direction.    
    '''
    
    x0 = z[0]
    y0 = f([s, x0])
    x1 = z[1]
    y1 = f([s, x1])
    x2 = z[2]
    y2 = f([s, x2])
    
    # similar to 1D case 
    x_min_y = (1/2) * ((x2**2 - x1**2) *y0 + (x0**2 - x2**2) *y1 + (x1**2 - x0**2) *y2) / ((x2 - x1) *y0 + (x0 - x2) *y1 + (x1 - x0) *y2)
    
    return x_min_y

# defining minimer in x direction 
def minimiser_x(f, guess1, guess2):

    '''Returns the minimum of a function, f(x,y) in the x direction while y is fixed.
       
    Parameters
    -----------------------        
        f : A function of 2 variable, f(x,y)
        guess1 (list) : The initial guess list of three x values (float)
        guess2 (float) : The initial guess value of y (NOT a list)
    
    Returns
    -----------------------
        x_min_ob_x (float) : The x that gives the minimum of f(x,y) in x direction.
    '''
    
    d_x = 0.5
    z0_x = guess1
    x_min_list_x = []
    l_min_list_x = []
    
    while d_x > 1e-6:       # similar to 1D case to the argument to f is modified
         
        z_l_x = []                          
        for i in z0_x:
            z_l_x.append(f([i, guess2]))
            
        x_m_0_x = x_min_x(f, z0_x, guess2)
        z0_x[z_l_x.index(max(z_l_x))] = x_min_x(f, z0_x, guess2)
        d_x = abs(x_min_x(f, z0_x, guess2) - x_m_0_x)
        x_min_list_x.append(x_min_x(f, z0_x, guess2))
        l_min_list_x.append(f([x_min_x(f, z0_x, guess2), guess2]))
    
    x_min_ob_x = x_min_list_x[l_min_list_x.index(min(l_min_list_x))]
    
    return x_min_ob_x

# similarly for y direction
def minimiser_y(f, guess1, guess2):
    
    '''Returns the minimum of a function, f(x,y) in the y direction while x is fixed.
       
    Parameters
    -----------------------        
        f : A function of 2 variable, f(x,y)
        guess1 (float) : The initial guess value of x (NOT a list)
        guess2 (list) : The initial guess list of three y values (float)

    Returns
    -----------------------
        x_min_ob_y (float) : The y that gives the minimum of f(x,y) in x direction.
    '''
    
    d_y = 0.5
    z0_y = guess2     
    x_min_list_y = []
    l_min_list_y = []
    
    while d_y > 1e-6:        # similar to 1D case to the argument to f is modified
        
        z_l_y = []                         
        for i in z0_y:
            z_l_y.append(f([guess1, i]))
            
        x_m_0_y = x_min_y(f, z0_y, guess1)
        z0_y[z_l_y.index(max(z_l_y))] = x_min_y(f, z0_y, guess1)
        d_y = abs(x_min_y(f, z0_y, guess1) - x_m_0_y)
        x_min_list_y.append(x_min_y(f, z0_y, guess1))
        l_min_list_y.append(f([guess1, x_min_y(f, z0_y, guess1)]))
    
    x_min_ob_y = x_min_list_y[l_min_list_y.index(min(l_min_list_y))]
    
    return x_min_ob_y
    
# minimisiing x and y one after the other
def univariate(f, guess):    
    
    '''Returns the minimum of a function, f(x,y).
       
    Parameters
    -----------------------        
        f : A function of 2 variable, f(x,y). Here f = likelihood_2d.
        guess (list) : A list of two lists; one for x guesses, one for y guesses

    Returns
    -----------------------
        (x_min, y_min) : The minimum point of f(x,y)
    '''
     
    x_guess = guess[0]        #list of three points for x (theta)        
    y_guess = guess[1]        #list of three points for y (m_squared)
    
    y_init = 0.0026           # a guess for the first step

    d1 = 0.1         #difference in x: theta
    d2 = 0.1         #difference in y: m_squared
    
    while d1 > 1e-5 or d2 > 1e-7:            # individual criteria
        
        # first minimisation in x
        theta_init = minimiser_x(likelihood_2d, x_guess, y_init)
        # first minimisation in y using the x_min previously
        m_init = minimiser_y(likelihood_2d, theta_init, y_guess)
        
        # second minimisation in x using y_min previously
        thet_2 = minimiser_x(likelihood_2d, x_guess, m_init)
        # second minimisation in y using the x_min previously
        m_2 = minimiser_y(likelihood_2d, thet_2, y_guess)
        print(thet_2, m_2, f([thet_2, m_2]))
                
        d1 = abs(thet_2 -  theta_init)   # calculate the subsequent differences
        d2 = abs(m_2 - m_init)
        
        y_init = m_2  
        
    print('The d1 is', d1, 'The d2 is', d2)   # to observe the difference
        
    return thet_2, m_2

#%%
    
# minimising first min
guess_uni1 = [[0.56, 0.7, 0.2], [0.003, 0.002, 0.0025]]
min1 = univariate(likelihood_2d, guess_uni1)  

# minimising second min
guess_uni2 = [[0.85, 0.975, 0.9], [0.003, 0.002, 0.0025]]
min2 = univariate(likelihood_2d, guess_uni2)  
        
print('The first global min is', round(min1[0], 3),',', round(min1[1], 6))    
print('The second global min is', round(min2[0], 3),',', round(min1[1], 6))    

#%%
# check using scipy routine
# importing scipy to check
import scipy.optimize as op

res_2d_1 = op.minimize(likelihood_2d, [0.6, 2.4e-3], method='nelder-mead')
print('The scipy result is', res_2d_1)
res_2d_2 = op.minimize(likelihood_2d, [0.9, 2.4e-3], method='nelder-mead')
print('The scipy result is', res_2d_2)

# theta =  0.67032787, 0.90054083  m_sq =  0.00259088 ---> works as expected

#%%        
        
#simultaneous minimisation --> gradient descent

def grad(f, x, y):
    
    '''Returns the gradient vector of a function, f at a given point (x,y).
       
    Parameters
    -----------------------        
        f : A function of 2 variables
        x (float) : The x of the point (x,y)
        y (float) : The y of the point (x,y)

    Returns
    -----------------------
        The gradient vector; first element is the gradient in x direction,
        second element is the gradient in y direction.
    '''
    
    delta_x = 1e-5       # the step size in x (the smaller, the more accurate)
    df_dx = ( f([ x + delta_x, y]) - f([x, y]) ) / delta_x
    
    delta_y = 1e-9       # the step size in y 
    df_dy = ( f([ x, y + delta_y]) - f([x, y]) ) / delta_y
    
    return np.array([df_dx, df_dy])

def grad_minimiser(f, guess, alpha, iterations):
    
    '''Returns the minimum of a function, f(x,y).
       
    Parameters
    -----------------------        
        f : A function of 2 variables, f(x,y).
        guess (list) : A list of two elements; one for x guess, one for y guess.
        alpha (array) : The step size in two directions; x and y
        iterations : Number of steps in gradient descent

    Returns
    -----------------------
        (x_min, y_min) : The minimum point of f(x,y)
        loss_list (list) : A list of f(x_min, y_min) at each step
    '''    
    a = alpha                   # step size in gradient descent
    max_it = iterations         # max num of iterations
    it = 0                      # initial iterations
    x_0 = guess  
    loss_list = []
    
    while it < max_it :           # run until max iterations
        
        # descent in the opposite direction of gradient
        x_0 = x_0 - a.T * grad(f, x_0[0], x_0[1])
        
        it = it + 1               # increase the counter
        loss_list.append(f(x_0))
        
        print(it, x_0, f(x_0))    # to observe each step
        
    return x_0, loss_list     
        

#%%
#validate the grad minimiser
    
def f_validate(u):
    
    '''Returns the value of validation function given a point u
    
    Parameters
    -------------------
        u (list) : Two input variables (x,y) into the function
    
    Returns:
    ------------------
        r (float) : The value of validation function at u
    '''
    x = u[0]
    y = u[1]
    r = 2*x**2 + x + 3*y**2 +3*y + 4
    return r
#%%
# plotting a contour plot for validation function
    
x_v = np.linspace(-10, 10, 100)
y_v = np.linspace(-10, 10, 100)
x_v, y_v = np.meshgrid( x_v, y_v)
r_map = f_validate([x_v, y_v])

fig_c2, ax_c2 = plt.subplots()
cs2 = ax_c2.contourf(x_v, y_v, r_map, 300,  cmap = 'jet')
cbar2 = fig_c2.colorbar(cs2)
ax_c2.set(title='Validation function',
           xlabel= 'x' ,
           ylabel=  'y') 
ax_c2.grid(False)
plt.show()

#%%
# running the grad minimiser on the validation fn

alpha1 = np.array([1e-1, 1e-1])                # step size
valid_min, valid_list = grad_minimiser(f_validate, [-0.2, 0], alpha1, 20)

# plotting f after each step --> to ensure convergence

ax30 = plt.figure(30).add_subplot(111)
ax30.set(title='Validation function after each step',
           xlabel= 'Number of step',
           ylabel='f',
           xlim = (-1,20))
    
step_valid = np.arange(0, len(valid_list))    
plt.plot(step_valid, valid_list, 'r')
plt.show()

print('The min of validation function is', valid_min)  #answer should be [-0.25, -0.5]

#%%

# running grad descent on nll_2d

alpha2 = np.array([1e-4, 1e-9])        # step size; different in each direction
g_vector = [0.6, 2.4e-3]
min_x1, loss_list_2d_1 = grad_minimiser(likelihood_2d, g_vector, alpha2, 20)    

g_vector2 = [0.9, 2.4e-3]    
min_x2, loss_list_2d_2 = grad_minimiser(likelihood_2d, g_vector2, alpha2, 20)    

# plotting the nll_2d at each step

ax21 = plt.figure(21).add_subplot(111)
ax21.set(title='NLL after each step',
           xlabel= 'Number of step',
           ylabel='NLL',
           xlim = (-1, 20))
    
step1 = np.arange(0, len(loss_list_2d_1))    
plt.plot(step1, loss_list_2d_1, 'r', label = 'First minimum')
plt.plot(step1, loss_list_2d_2, 'b', label = 'Second minimum')
plt.legend()
plt.show()

print('The first minimum is', min_x1)   
print('The second minimum is', min_x2)

#%%

# uncertainty in 2d minimisation

def error_2d(f, min_point, axis):
    
    '''Returns the error of a function f(x,y) at a given minimum point,
      (x_min, y_min) for a given axis.
    
    Parameters
    -------------------
        f : A function of two variables f. Here f = likelihood_2d
        min_point (list) : The minimum point [x_min, y_min] of f
        axis (str) : The direction in which the error to be determined
        
    Returns
    ----------------
        std (float) : The standard deviation or the error    
    '''
    
    theta_min = min_point[0]
    m_min = min_point[1]
    l_min = f(min_point)
    l_plus = l_min + 0.5
    
    # similar to the 1D case
    
    if axis == 'theta':       # if the direction is in theta
        d = 0.1
        criteria = 0.01
        x_range = np.linspace(theta_min - d, theta_min + d, 1000)
        err_list = []
        
        for i in x_range:
            if (abs(f([i, m_min]) - l_plus) < criteria):
                err_list.append(theta_min - i)
        err_plus = max(err_list)
        err_minus = abs(min(err_list))  

    if axis == 'm':           # if the direction is in theta
        criteria = 0.01
        m_range = np.linspace(0.002, 0.003, 1000)
        err_list = []
        
        for i in m_range:
            if (abs(f([theta_min, i]) - l_plus) < criteria):
                err_list.append(m_min - i)
        
        err_plus = max(err_list)
        err_minus = abs(min(err_list))  
   
    std = (err_minus + err_plus) /2
    return std

#%%
# finding the error in  both directions
    
err_vec1 = [error_2d(likelihood_2d, min_x1, 'theta'), error_2d(likelihood_2d, min_x1, 'm')]
err_vec2 = [error_2d(likelihood_2d, min_x2, 'theta'), error_2d(likelihood_2d, min_x2, 'm')]

##quoting the results

print('The first theta minimum is', round(min_x1[0], 3), '+-', round(err_vec1[0], 3))
print('The first m minimum is', round(min_x1[1], 6), '+-', round(err_vec1[1], 6))
print('The second theta minimum is', round(min_x2[0], 3), '+-', round(err_vec2[0], 3))
print('The second m minimum is', round(min_x2[1], 6), '+-', round(err_vec2[1], 6))
