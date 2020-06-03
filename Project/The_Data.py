'''
Computational Physics Project
A log-likelihood fit for extracting neutrino oscillation parameters

Part 1: The Data

##Click 'Run file' to display all 4 plots at one go##
'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# to read the data.csv file
def read_file(name):
    return np.loadtxt(name, delimiter = ',', skiprows = 1 , unpack = 1)

# defining the observation (events) and simulated unoscillated flux(sim) from data file
events, sim = read_file('data.csv')
                                             

# defining the parameters for plotting graphs
params = {
       'axes.labelsize': 12,
       'font.size':12,
       'legend.fontsize': 11,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'figure.figsize': [6, 4],
       'figure.dpi': 100,
       'axes.grid': 1,
        }
plt.rcParams.update(params)

if __name__ == '__main__':      # only execute in this file, not while imported
    
    # plotting the histogram of observed events --> used step plot here
    E_step = np.linspace(0, 10, 200)      # defining x-range for step plot
    fig = plt.figure(1)    
    ax = fig.add_subplot(111)
    ax.set(title='Muon Nuetrino Event Rate',
           xlabel='Energy (GeV)',
           ylabel='Event Rate')    
    plt.step(E_step, events, color = 'red')
    #plt.savefig('1')
    plt.show()

#%%
# defining oscillation probability, p
def p(theta, m_squared , L, E):
    
    '''Returns the oscillation probability of a muon neutrino staying as a muon neutrino.
    
    Parameters
    -----------------
        theta (float) : The mixing angle (rad)
        m_squared (float) : The difference of squared masses of two neutrinos (eV^2)
        L (float) : The distance travelled by the neutrino (km)
        E (float) : The energy of the neutrino (GeV)
    
    Returns
    ----------------
        The oscillation probability (float) in the range [0,1].
    '''
    
    arg = 1.267 * m_squared * L / E         # argument of the second sin^2
    
    prob = 1 - ((np.sin(2 * theta))**2) * ((np.sin(arg))**2)
    return prob


if __name__ == '__main__':
    
    # plotting oscillation probability for different values of parameters

    L_initial = 295                              # fixed
    m_squared_initial = [0, 1e-3, 2.4e-3, 3e-3]  # different m_squared values
    theta_initial = [0.2, 0.4, 0.785, 1]         # different theta values

    E_range = np.linspace(0.01, 2, 100)             # Energy in x-range


    ax3 = plt.figure(3).add_subplot(111)

    # varying m_squared, then plotting
    for i in m_squared_initial:
        p_initial = p(theta_initial[2], i, L_initial, E_range)
        plt.plot(E_range, p_initial, label = '$\Delta {m^2}_{23}$ = %r' %i) 
        
    ax3.set(title='Varying the $\Delta {m^2}_{23}$',   # title is omitted in the report
                xlabel='Energy (GeV)',
                ylabel='Oscillation Probability') 
    plt.legend() 
    plt.show()


    ax4 = plt.figure(4).add_subplot(111)

    # varying theta, then plotting
    for i in theta_initial:
        p_initial = p( i, m_squared_initial[2], L_initial, E_range)
        plt.plot(E_range, p_initial, label = r'$\theta_{23}$ = %r' %round(i, 3))
        
    ax4.set(title='Varying the $\Theta_{23}$',
           xlabel='Energy (GeV)',
           ylabel='Oscillation Probability') 
    plt.legend()  
    plt.show()

#%%
# defining expected event rate, lambda
x = np.arange(0.025, 10, 0.05)      # Energy as the mindpoints of the histogram


def lamb(u):
    
    '''Returns the expected event rate in each bin using the parameters, u.
    
    Parameters
    ---------------
        u (list): A list of parameters; theta and m_squared
        
    Returns
    --------------
        A list containing 200 values (float) of expected event rate.   
    
    '''
           
    L = 295                      # fixed distance
    theta = u[0]                 # first element of u
    m_sq = u[1]                  # second element of u
    
    lamb_list = []               #initialise an empty list
    for i in range(200):      
        
        # calculate the epected event rate in each bin, there are 200 bins
        
        lamb = sim[i] * p(theta, m_sq, L, x[i]) 
        lamb_list.append(lamb)                       #append in a list
    
    return lamb_list

if __name__ == '__main__':
    
    u_init = [np.pi/4, 2.4e-3]                        # initial parameters

    # plotting event rates

    ax5 = plt.figure(5).add_subplot(111)
    ax5.set(title='Adjusted Event Distribution',
           xlabel='Energy (GeV)',
           ylabel='Event Rate',
           xlim = (0,5)) 

    plt.step(E_step, events, color = 'red', label = 'Observation')
    plt.step(E_step, sim, color = 'blue', label = 'No oscillation')
    plt.step(E_step, lamb(u_init), color = 'green', label = 'Adjusted')
    plt.legend() 
    plt.show()

