'''
Computational Physics Project
A log-likelihood fit for extracting neutrino oscillation parameters

Part 2: 1D Minimisation
'''

# importing variables and functions from previous .py file
from The_Data import events, sim, p, x, np, plt

# defining NLL in 1D

def likelihood(theta):
    
    
    '''Returns the Negative Log likelihood of the events for a given theta.
       The other parameters (L, m_squared) are fixed. 
    
    Parameters
    ----------------
        Theta (float) : The mixing angle
        
    Returns
    ----------------
        The Negative Log likelihood (float)   
    
    '''
    
    L = 295                              # fixed distance
    m_sq = 2.4e-3                        # fixed m_sauared in the 1-D case
    nll = []                             # initialise an empty list for nll
    k = events                           # the events per bin from data
    
    for i in range(200):
        
        if k[i] > 0:                     # to avoid log(0)
            
            lamb = sim[i] * p(theta, m_sq, L, x[i])
            nll_i = lamb - k[i] + (k[i] * np.log( k[i] / lamb))
            nll.append(nll_i)
            
        if k[i] == 0:                   # ignore the log term
            lamb_0 = sim[i] * p(theta, m_sq, L, x[i])
            nll.append(lamb_0)
    
    return sum(nll)                     # sum of the elements in the nll list


    
# plotting nll with respect to theta
    
thet = np.linspace(0, np.pi/2,1000)
 
ax6 = plt.figure(6).add_subplot(111)
ax6.set(title='NLL against $\Theta_{23}$ with a fixed $\Delta {m^2}_{23}$',
         xlabel= r"$\theta_{23}$ (rad)",
         ylabel='NLL')  
plt.plot(thet, likelihood(thet), color = 'red')
plt.show()

#%%
#building a parabolic minimiser

def x_min(f,z):
    
    '''Returns the minimum point of a parabolic function that contains the three
       points in the list of z. This is to be used in minimiser() later.
       
    Parameters
    --------------------------
        f : A function of 1 variable that returns f (float) at the three x's
        z (list) : A list of three x values (float)
    
    Returns
    --------------------------
        x_min (float) : The x that gives the minimum of a parabolic function
        
    NOTE: f is not necessarily parabolic, but is nearly parabolic at its minimum
    '''
    
    x0 = z[0]      # first x
    y0 = f(x0)     # f of first x
    x1 = z[1]      # second x
    y1 = f(x1)     # f of second x
    x2 = z[2]      # third x
    y2 = f(x2)     # f of third x
    
    # using Langrage polynomials
    
    x_min = (1/2) * ((x2**2 - x1**2) *y0 + (x0**2 - x2**2) *y1 + (x1**2 - x0**2) *y2) / ((x2 - x1) *y0 + (x0 - x2) *y1 + (x1 - x0) *y2)
    
    return x_min

def minimiser(f, guess):
    
    '''Returns the minimum of a function, f(x) along with the list of minimum of f
       at each step of minimisation.
       
    Parameters
    -----------------------        
        f : A function of 1 variable, f(x)
        guess (list) : The initial guess list of three x values (float)
    
    Returns
    -----------------------
        x_min_ob (float) : The x that gives the minimum of f(x)
        l_min_list (list) : The list of minimum values of f at each step of
                            minimisation.    
    '''
    d = 0.5               # initial value for the difference in x_min
    z0 = guess            # redifining for simplicity
    x_min_list = []       # initialising an empty list for x_min's
    l_min_list = []
    
    while d > 1e-5:       # run until the differene is small enough
        
        z_l = []
        for i in z0:                # for each x in guess, append its f(x)
            z_l.append(f(i))
            
        x_m_0 = x_min(f, z0)        # minimise f using the guess
        
        # replace the x that gives the highest f with the x_min
        z0[z_l.index(max(z_l))] = x_min(f, z0)
        
        d = abs(x_min(f, z0) - x_m_0)      # find new x_min, then calculate the difference
        x_min_list.append(x_min(f, z0))       # append the x_min's
        l_min_list.append(f(x_min(f, z0)))    # append the f(x_min)'s
        #print('The x_min is',x_min(f, z0), 'and f(x_min) is', f(x_min(f, z0)))  # prints x_min and f(x_min) at each step
    
    # choose the x_min that gives the lowest f(x_min) in the list
    x_min_ob = x_min_list[l_min_list.index(min(l_min_list))]  
    
    return x_min_ob, l_min_list


#checking the minimiser works with other function - using p_mod (modified)

def p_mod(E):
    
    '''Returns the oscillation probability for a given E.
       Modified the previous p to set to include only one variable
    
    Parameters
    ----------------
        E (float) : Energy of neutrino
        
    Returns
    ----------------
        The oscillation probability (float)  
    '''
    
    theta = 0.785                # fixed 
    m_squared = 2.43e-3          # fixed
    L = 295                      # fixed
    arg = 1.267 * m_squared * L / E      
    prob = 1 - ((np.sin(2 * theta))**2) * ((np.sin(arg))**2)
    return prob


    
guess_test = [0.6, 0.7, 0.65]               # initial guess
p_min, p_l = minimiser(p_mod, guess_test)   # run the minimiser on P
p_l = [x*1e5 for x in p_l]                  # rescale for plotting
ax22 = plt.figure(7).add_subplot(111)
ax22.set(title='P against number of step',
           xlabel= 'Number of step',
           ylabel=r'P ($ \times 10^{-5}$)')
    
step2 = np.arange(0, len(p_l))             # x_range for steps 
plt.plot(step2, p_l, 'r') 
plt.show()
print('The min of p is', round(p_min, 5))  # prints the final answer

 
# to validate with scipy routine
    
#op.minimize(p_mod, 0.67, method='nelder-mead')  ---> gives [0.57820215]

#it works as expected

#%%

# running the minimiser on the likelihood function, NLL
    
# first min    
    
guess1 = [0.56, 0.7, 0.2]                  # initial guess for first min
theta_min, l_2 = minimiser(likelihood, guess1)      # run the minimiser on NLL

# second min
guess2 = [0.850, 0.9, 0.975]               # initial guess for first min
theta_min2, l_3 = minimiser(likelihood, guess2)      # run the minimiser on NLL

# plotting loss after step for both min

ax23 = plt.figure(7).add_subplot(111)
ax23.set(title='NLL against $x_{min}$ after each step',
           xlabel= 'Number of step',
           ylabel=r'NLL')  
    
step3 = np.arange(0, len(l_2))   
plt.plot(step3, l_2, 'r', label = 'First minimum') 
   
step4 = np.arange(0, len(l_3))    
plt.plot(step4, l_3, 'b', label = 'Second minimum')
plt.legend()
plt.show()

print('The x_min_1 is', round(theta_min, 6))
print('The x_min_2 is', round(theta_min2, 6))

print(l_2)
#%%
# solving using scipy optimizer to check answer
import scipy.optimize as op

guess_sci = 0.66
res_p = op.minimize(likelihood, guess_sci , method='nelder-mead')

guess2_sci = 0.9
res_p_2 = op.minimize(likelihood, guess2_sci , method='nelder-mead')

#print(res_p, res_p_2)         # x_min_1 = 0.66264258 , x_min_2 = 0.90817383
# it checks out

#%%
# accuracy of parameter, theta

# using 0.5 difference method from NLL eq.

# calculating for first minimum

thet_2 = np.linspace(0.61, 0.8,1000)      # defining a list of theta with high density
l_min = likelihood(theta_min)             # defining a list of nll(theta) for above theta's
l_plus = l_min + 0.5                 
err_list =[] 

for i in thet_2:                              # find theta that gives l_plus
    if (abs(likelihood(i) - l_plus) < 0.01):
        err_list.append(theta_min - i)
        
err_plus = max(err_list)               # max is the error in plus direction
theta_plus = theta_min + err_plus
err_minus = abs(min(err_list))         # min is the error in minus direction
theta_minus = theta_min - err_minus 
print('The error_minus 1', err_minus, 'and error_plus 1', err_plus) 
    

#checking for second min

thet2_2 = np.linspace(0.85, 0.95,1000)       # covering the second minimum 
l_min2 = likelihood(theta_min2)
l_plus2 = l_min2 + 0.5
err_list2 =[]

for i in thet2_2:                             # find theta that gives l_plus
    if (abs(likelihood(i) - l_plus2) < 0.01):
        err_list2.append(theta_min2 - i)
        
err_plus2 = max(err_list2)
theta_plus2 = theta_min2 + err_plus2
err_minus2 = abs(min(err_list2))  
theta_minus2 = theta_min2 - err_minus2 
print('The error_minus 2', err_minus2, 'and error_plus 2', err_plus2)



# next, using 0.5 difference method from parabolic eq.

# firstly need to find the last parabolic estimate

d = 0.5                     # initialising difference d
z0 = [0.56, 0.7, 0.2]       # initial guess for three points

while d > 1e-5:      # run the same minimisation but this time, 
                     # get the three points that give the final x_min
                                  
    z_l = []
    for i in z0:
        z_l.append(likelihood(i))
            
    x_m_0 = x_min(likelihood, z0)
    z0[z_l.index(max(z_l))] = x_min(likelihood, z0)
    d = abs(x_min(likelihood, z0) - x_m_0)

last_parab_list = z0          # the last parabolic estimate
    

def parabola(f, x, s):
    '''Returns a parabolic function that fits three given points (x, f(x))
       using Langrage Polynomials.
    
    Parameters
    ----------------------
        f : A function that gives f(x) at a given x
        x (float) : Input variable for the parabolic function
        s (list) : The list of three points to be fitted
    
    Returns
    ---------------------
        A parabolic function 
    '''
    # defining each term for simplicity
    
    a1 = (x - s[1]) * (x - s[2]) * f(s[0]) / ((s[0] - s[1]) * (s[0] - s[2]))
    b1 = (x - s[0]) * (x - s[2]) * f(s[1]) / ((s[1] - s[0]) * (s[1] - s[2]))
    c1 = (x - s[0]) * (x - s[1]) * f(s[2]) / ((s[2] - s[0]) * (s[2] - s[1]))
    
    return a1 + b1 +c1


l_min_para = parabola(likelihood, theta_min, last_parab_list)  # min value of parabola at x_min
l_plus_para = l_min_para + 0.5
err_list_para =[]

for i in thet_2:                      # same procedure as nll above
    
    if (abs(parabola(likelihood, i, last_parab_list) - l_plus_para) < 0.01):
        err_list_para.append(theta_min - i)
        
err_plus_para = max(err_list_para)
theta_plus_para = theta_min + err_plus_para
err_minus_para = abs(min(err_list_para))  
theta_minus_para = theta_min - err_minus_para 
print('The error from parabola for first min', err_minus_para, err_plus_para)    

# it gives exactly same answer as NLL method

#%%

# defining automatic method to give uncertainty

def error(f, x_min):
    '''Returns the error of a function at a given minimum point, x_min.
    
    Parameters
    -------------------
        f : A function 
        x_min (float) : The minimum x of f(x)
        
    Returns
    ----------------
        std (float) : The standard deviation or the error    
    '''
    
    x_range = np.linspace(x_min - 0.1, x_min + 0.1, 1000)
    l_min = f(x_min)                        #using likelihood function
    err_list = []
    l_plus = l_min + 0.5
    
    for i in x_range:
        if (abs(f(i) - l_plus) < 0.01):
            err_list.append(x_min - i)
    err_plus = max(err_list)
    err_minus = abs(min(err_list))  
    
    std = (err_minus + err_plus) /2         # taking the average
    
    return std

       
std1 = error(likelihood, theta_min)
std2 = error(likelihood, theta_min2) 

#quoting the full result

print('The first minimum of theta is', round(theta_min, 3), '+-', round(std1, 3))
print('The second minimum of theta is', round(theta_min2, 3), '+-', round(std2, 3))
