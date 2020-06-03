'''
COMPUTATIONAL PHYSICS - ASSIGNMENT 
QUESTION 5 : RANDOM NUMBERS
'''

#importing appropriate libraries
import numpy as np
import matplotlib.pyplot as plt

#a)
'''
We use random number generator class in numpy called RandomState which creates
a container for the Mersenne Twister pseudo-random number generator.
We want to create uniformly generated random numbers in range (0,1)

'''

# uniform deviate generator
r = np.random.RandomState(seed = 1)    #random number generator class with seed = 1

rand_uniform = []       #initialising an empty list
N = 100000              #number of random numbers needed

# create N uniform random numbers
for i in range(N):
  x_rand = r.uniform(0,1)     
  rand_uniform.append(x_rand)

#histogram of bin 10
plt.figure(1)
plt.title('Histogram of bin 10')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.hist(rand_uniform, bins = 10, color = 'red')
plt.axhline(y=10000, color='blue', linestyle='dashed')  #horizontal line for comparison
plt.show()

#histogram of bin 100
plt.figure(2)
plt.title('Histogram of bin 100')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.hist(rand_uniform, bins = 100, color = 'red')
plt.axhline(y=1000, color='blue', linestyle='dashed')  #horizontal line for comparison
plt.show()

#%%
#b
'''
We want to compute random numbers distributed over the interval of (0, np.pi)
with pdf = (1/2) cos (x/2).
We use transformation method. 
We find the cdf by integrating pdf from 0 to x.
Then, we take the inverse the cdf to give the pdf_new. 
We can map each x given by uniform deviate into new y which has pdf_new.

'''
#defining the pdf for plotting
def pdf_1(x):
  a = 3100             #amplitude correction according to the bin number chosen
  return a*(1/2)*np.cos(x/2)

r = np.random.RandomState(seed = 1)  #random number generator class with seed = 1

def rand_1(N):
    
    '''
    Input: Number of random numbers needed to be generated, N
    Output: A list of N random numbers with pdf_1
    Procedure:
        Intialises an empty list
        For N times:
            A point y is uniformly generated
            Run y into the new pdf (inversed cdf)
            Add the result into the list    
    '''
    
    rand_pdf_1 =[]
    for i in range(N):
        y_uni = r.uniform(0,1)
        y_rand_uni = 2*np.arcsin(y_uni)    #put into inversed cdf, which is the new pdf
        rand_pdf_1.append(y_rand_uni)
    return rand_pdf_1

rand_b = rand_1(N)      #creating 10^5 samples 

#histogram of bin 100
plt.figure(3)
plt.xlabel('y')
plt.ylabel('Frequency')
plt.hist(rand_b, bins = 100, color = 'red')

x = np.linspace(0, np.pi, 100)                        #for plotting
plt.plot(x, pdf_1(x), color = 'blue', label = 'cos(x/2)', linestyle='dashed' )  #for comparison
plt.legend()
plt.show()

#%%

#c)
'''
We want to generate random number with given pdf_2 using rejection method.
We need a comparison function, C which is always bigger than pdf for all z in the interval. 
pdf 1 in the previous part is a good comparison function, but we alter its amplitude.
Firstly we pick y_i distributed with pdf_1 and compute a uniform random number p_i
in the range of (0, C(y_i)).
If pdf_2(y_i) is bigger than p_i, we add y_i to a list, otherwise reject.

To ensure we have around 10^5 samples, we calculate the ratio of areas of pdf_1 to pdf_2
since areas correspond to the probability. The ratio of area is 4/np.pi which 
is around 1.27323.

'''
N2 = 127323                       #This N2 will give around N random numbers
rand_pdf_1_mod = rand_1(N2)       #we use N2 of pdf_1 numbers

#defining pdf_2
def pdf_2(x):
  return (2/np.pi)*((np.cos(x/2))**2)

#defining comparison function
def c_y(x):
    return (2/np.pi)*np.cos(x/2)

rand_pdf_2 = []
r = np.random.RandomState(seed = 1)   #random number generator class with seed = 1

for i in range(N2):                 #runs the algorithm described
     y_i = rand_pdf_1_mod[i]
     p_i = r.uniform(0,c_y(y_i))         
     if pdf_2(y_i) > p_i:
         rand_pdf_2.append(y_i)

c = 3000              #arbitrary amplitude correction according to the bin number       chosen
k = 1.228             #arbitrary amplitude correction according to the bin number chosen

#histogram of bin 100
plt.figure(4)
plt.xlabel('z')
plt.ylabel('Frequency')
plt.hist(rand_pdf_2, bins = 100, color = 'orange', linestyle='dashed')

plt.plot(x, c*pdf_2(x), color = 'red', label = '$cos^2(x/2)$', linestyle='dashed')
plt.plot(x, k * pdf_1(x), color = 'black', label = '$cos(x/2)$', linestyle='dashed')
plt.legend()
plt.show()

len_2 = len(rand_pdf_2) 
print(len_2)          #to see if we get around 10^5 samples - we get 100018

#%%
#calculating the time taken for transformation and rejection method and its ratio

import timeit

code_to_test_1 = '''

rand_b = rand_1(N) 

'''

elapsed_time_1 = timeit.timeit(code_to_test_1, number=10, globals =globals())/10
print('runtime for transformation method',elapsed_time_1)

code_to_test_2 =  '''

rand_pdf_2 = []
r = np.random.RandomState(seed = 1)   #random number generator class with seed = 1

for i in range(N2):                   #runs the algorithm described
     y_i = rand_pdf_1_mod[i]
     p_i = r.uniform(0,c_y(y_i))         
     if pdf_2(y_i) > p_i:
         rand_pdf_2.append(y_i)
'''

elapsed_time_2 = timeit.timeit(code_to_test_2, number=10, globals =globals())/10
print('runtime for rejection method',elapsed_time_2)

r = elapsed_time_2/elapsed_time_1
print('The ratio of time taken in rejection method to transformation method is', r)
