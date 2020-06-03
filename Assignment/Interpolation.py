'''
COMPUTATIONAL PHYSICS - ASSIGNMENT 
QUESTION 3 : INTERPOLATION
'''

#importing appropriate libraries
import numpy as np
import matplotlib.pyplot as plt


#a)
'''
We want to  interplot a tabulated set of x and y linearly.

'''

# creating an arbitary table of data 
x_inter = np.array([1.5, 2.3, 5.2, 6.9, 7.1])
y_inter = np.array([2.5, 5.0, 7.8, 4.2, 3.5])

plt.scatter(x_inter, y_inter)

#defining a function that interplots linearly
def lin_inter(x, y):
  '''
  Input: Set of x and y data
  Output: A plot containing x and y data, linearly interpolated
  Procedure: 
      For each pair of consecutive points, a straight line needs to be plotted.
      Hence, we need the constants in a straght line equation (e.g. first derivative). 
      We calculate these using existing formula for each pair. 
      A line equation is defined and plotted.      
  
  '''
        
  for i in range(len(x)-1):   #need 4 lines of interplotion for a set of 5 points
      
    def f_i(x_f):
      return ((x[i+1] - x_f)*y[i] + (x_f - x[i])*y[i+1]) / (x[i+1] - x[i])
      
    x_f = np.linspace(x[i],x[i+1], 50)    #defined between each pair of points
    line = f_i(x_f)
    
    plt.plot(x_f, line,color = 'red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

plt.figure(1)        
lin_inter(x_inter,y_inter)

#%%

#b)
'''
We want to perform a cubic spline interpolation on the set of data above.
For cubic spline we need three points at once. The first and second derivatives
of those points will be considered. 
We can use the fundamental expression given in the notes. 
We then need to solve (n-1) eq. with  (n+1) unknowns. So, we set second derivatives
at ends to be zero, thus called 'natural'. 
This reduces the no. of unknowns to (n-1). 
These simultaneous equations for second derivates for all points is set up as a 
matrix eq. 
We can use the matrix solver functions from previous methods to solve for all
second derivatives. Along with the first derivatives (similar to linear interpolation), 
we could get the equation of curve for a set of three points.

'''

###############################################################################
#copy pasting the functions from matrix methods

#lu () to be used later in cubic spline function
def lu(mat):
    
  '''
  Input: An N x N matrix (dimension N)  
  Output: 
      A lower triangular matrix, L
      An upper triangular matrix, U
  Procedure: 
      The input matrix is decomposed into L and U using Crout's method. 
      
  '''
    
  n = len(mat)          #dimension of input matrix
  L = np.zeros((n,n))   #initialise a zero matrix for L
  U = np.zeros((n,n))   #initialise a zero matrix for U
  for i in range(n):
      
    L[i][i] = 1         #initialise the diaganol entries of L to be 1
    
    for k in range(i,n):      
        #for each i = 0,1,..,n-1 , U[i,k] is solved 
        sum0 = sum(L[i][j] * U[j][k] for j in range(i)) 
        U[i][k] = mat[i][k] - sum0
    
    for k in range(i,n):
        #for each i = 0,1,..,n-1 , L[k,i] is solved
        sum1 = sum(L[k][j] * U[j][i] for j in range(i))
        L[k][i] = (mat[k][i] - sum1) / U[i][i] 
        
  return L, U

#solve() to be used in cubic spline function
def solve(L, U, b):
  '''
  Input:
      An N x N lower triangular matrix, L
      An N x N upper triangular matrix, U
      An N x 1 vector, b
  Output: 
      An N x 1 vector, x
  Procedure:
      y and x are initialised with zero entries.
      Use forward substitution to solve y
      Use backward substitution to solve x
      
  '''
  n = len(L)                #dimension of L or U
  y = np.zeros(n)           #initialise y
  x = np.zeros(n)
  
  #forward substitution
  y[0] = b[0]/ (L[0][0])
  for i in range(n):
    sum3 = sum(L[i][j] * y[j] for j in range(i))
    y[i] = (b[i] - sum3) / (L[i][i])
  
  #y is solved/known
  #backward substitution
  x[n-1] = y[n-1] / (U[n-1][n-1])
  for i in range(n-1,-1, -1): 
    sum4 = sum(U[i][j] * x[j] for j in range(i+1,n))
    x[i] = (y[i] - sum4) / (U[i][i]) 
  return x

###############################################################################
  
#defining a function to perform cubic spline interpolation
def cubic_spline(x, y):
    
  '''
  Input: Set of x and y data 
  Output: A plot containing x and y data, interpolated using cubic spline
  Procedure:
      Each coefficient of second derivatives in the fundamental expression is 
      defined as alpha for (i-1)th, gamma for ith and beta (i-1)th coefficient. 
      The RHS of fundamental expession is given by rho. 
      These coefficient are defined and filled in a matrix. 
      LU decomposing the matrix and solving it for second derivatives.
      For (i-1)th, i and (i+1)th points, a curve is defined and plotted by using
      first and second derivatives.     
      
  '''
  
  n = len(x) - 1           #dimension of derivative matrix
  alpha = np.zeros(n)      #initialising the coeffecients
  beta = np.zeros(n)
  gamma = np.zeros(n)
  rho = np.zeros(n)

  D = np.zeros((n,n))  #initialisiong derivative matrix

  #build a matrix
  for i in range(n):
    #finding the elements of matrix 

    alpha[i] = (x[i] - x[i-1]) /6
    beta[i] = (x[i+1] - x[i-1]) /3
    gamma[i] = (x[i+1] - x[i]) /6
    rho[i] = ((y[i+1] - y[i]) / ( x[i+1] - x[i])) - ((y[i] - y[i-1]) / (x[i] - x[i-1]))
  
  for j in range(n):
    #filling up the matrix

    D[j][j] = beta[j]
    if j < n-1:
      D[j+1][j] = alpha[j+1]
      D[j][j+1] = gamma[j]
  
  L_der, U_der = lu(D)                #lu decomposing derivative matrix using previous method from Q2
  f_der = solve(L_der, U_der, rho)    #solving for second derivatives using matrix solver from Q2

  #building function for each (i-1, i and i+1 points) section

  for i in range(len(x)-2):   

    def f_cubic(x_cubic): 
      A_x = (x[i+1] - x_cubic) / (x[i+1] - x[i])   #coefficients
      B_x = (x_cubic - x[i]) / (x[i+1] - x[i])
      C_x = (1./6) * ((A_x)**3 - A_x) * (x[i+1] - x[i])**2
      D_x = (1./6) * ((B_x)**3 - B_x) * (x[i+1] - x[i])**2
    
      return A_x*y[i] + B_x*y[i+1] + C_x*f_der[i] + D_x*f_der[i+1]  #resultant function
      
    x_cubic = np.linspace(x[i],x[i+1], 100)    #high density to make it more continuous and smooth
    cubic_curve = f_cubic(x_cubic)
    plt.plot(x_cubic, cubic_curve, color = 'green')

plt.scatter(x_inter, y_inter)       #scatter plot
cubic_spline(x_inter, y_inter)      #cubic spline
lin_inter(x_inter,y_inter)          #linear 
plt.show()

#%%

#c)
'''
We want to perform linear and cubic spline interpolation on a given set of x and y.
We use the existing functions from previous parts.

'''

#defining the given data in an array
x_inter2 = np.array([-2.1, -1.45, -1.3, -0.2, 0.1, 0.15, 0.9, 1.1, 1.5, 2.8, 3.8, 3.80001])
y_inter2 = np.array([0.012155, 0.122151, 0.184520, 0.960789, 0.990050, 0.977761, 0.422383, 0.298197, 0.105399, 3.936690e-4, 5.355348e-7, 5.3552e-7])

plt.scatter(x_inter2, y_inter2)       #scatter plot
cubic_spline(x_inter2, y_inter2)      #cubic spline interpolation
lin_inter(x_inter2, y_inter2)         #linear interpolation
plt.show()

