'''
COMPUTATIONAL PHYSICS - ASSIGNMENT 
QUESTION 2: MATRIX METHODS
'''

#importing appropriate libraries
import numpy as np

#a)
'''
We want to LU decompose an arbitrary N x N matrix. 
LU decomposition is where a square matrix is decomposed into a lower
triangular matrix, L and upper triangular matrix, U; both with same dimension as
the initial matrix.  

'''

#create an arbitrary N x N matrix
M = np.array([[60.0, 30.0 ,20.0], [30.0, 20.0, 15.0], [ 20.0, 15.0, 12.0]])

#defining a function to perform lu decomposition
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
        
L, U = lu(M)   #performing lu decomposition on matrix M
print(L,U)
np.dot(L, U)   #L.U should give back the original matrix - it does

#%%

#b)
'''
We want to LU decompose a given matrix A and find its determinant.
'''
#defining the matrix A
A = [ [3, 1, 0, 0, 0],
      [3, 9, 4, 0, 0],
      [0, 9, 20, 10, 0],
      [0, 0, -22, 31, -25],
      [0, 0, 0, -55, 61] ]
  
low, upp = lu(A)     #performing lu decomposition on matrix M
print('L of A', low)
print('')            #make it more visible
print('U of A', upp) 
print('')            #make it more visible

print(np.dot(low, upp)) #checks out

#calculating the determinant
def det(mat):
    '''
    Input: An N x N matrix
    Output: The determinant of the input matrix
    Procedure: 
        The input matrix is decomposed into L and U using lu()
        The determinant is the product of the diagonal entries of upper
        triangular matrix, U.
    
    '''
    n = len(mat)
    lower, upper = lu(mat)     #performing lu decomposition
    product = 1                #initialising the product as 1
    for i in range(n):         
        product *= upper[i][i]  #multiplying the diagonal entries
    return product

print('The determinant of A is',det(A))

#np.linalg.det(A) #to check answer using numpy  - checks out

#%%

#c)

'''
We want to solve the matrix equation (LU)x = b 
(i.e. find vector x for a given L, U and vector b)
We can write (L.U)x = L.(U.x) = b
By letting U.x = y, we solve L.y = b for y.
Since L is a lower triangular matrix, we use forward substitution to find y.
We then use y to solve U.x = y for x.
Since U is an upper triangular matrix, we use backward substitution to find x.

'''

#defining a function to solve the matrix eq.
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

#%%

#d)
'''
We want to solve the matrix equation Ax = b.
We use the function from previous part
'''

#defining b
b = [2, 5, -4, 8, 9]

low, upp = lu(A)              #finding L and U
x_sol = solve(low , upp, b)   #using the method in previous part

print('The solution is', x_sol)

print(np.dot(A, x_sol)) #if the routine is correct, it should return b - checks out

#%%

#e)
'''
We want to find the inverse of a given matrix.
We know A = L.U and A_inv = U_inv.L_inv. So we find U_inv and L_inv separately.
We use the existing 'solve' function to find x for L.x[i] = I[i] where I[i] is 
the ith column of an identity matrix, I. 
We then concanate the x[i]'s to produce a matrix of x which is the L_inverse.
This is repeated with U. 
In 'solve' function, the first entry is used for the matrix we wanna solve (L or U), 
the second entry is used for identity matrix, as they are already in LU form, and the
final entry is the ith column of identity matrix.

'''

#defining function to solve inverse
def solve_inv(A):
   '''
   Input: An N x N matrix
   Output: The inverse of input matrix
   Procedure:
       The input matrix is LU decomposed.
       The inverse of L is found using the 'solve' function. 
       The inverse of U is found.
       The inverses of L and U are multiplied
       
   '''
      
   L,U = lu(A)             #lu decomposing the matrix
   n = len(A)
   b_new = np.identity(n)  #initialising the identity matrix
   l_inv = []              #initialising empty list for L_inverse and U_inverse
   u_inv = []
   
  #solve L_inv 
   for i in range(n):
     l_inv_col = solve(L, b_new, b_new[i])   #solved for each identity column
     l_inv.append(l_inv_col)                 #combine the lists
     
   L_inv = np.asarray(l_inv).T        #list converted into an array, transposed to make it an array of columns


   for i in range(n):
     u_inv_col = solve(U.T ,b_new, b_new[i])  #U is transposed to make it similar to L
     u_inv.append(u_inv_col)                  

   U_inv = np.asarray(u_inv)                  #the array is not transposed to give a valid A_inv

   A_inv = np.dot(U_inv, L_inv)
   return A_inv


A_inv = solve_inv(A)   #inverse of A

print('The inverse of A is', A_inv)

#b_new = np.identity(5)         #to be used for method below
#print(np.linalg.solve(A, b_new)) #to check the answer with np.linalg.solve method - checks out
#np.dot(A, A_inv)