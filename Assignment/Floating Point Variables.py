'''
COMPUTATIONAL PHYSICS - ASSIGNMENT 
QUESTION 1 : FLOATING POINT VARIABLES
'''

#a)
'''
Brief explanation:
    
Machine accuracy is meant to be the smallest number understood or recognised 
by the machine. 
In other words, whats the smallest number that you can add to a number and the
answer is not that number you started with? 
e.g. 1 + x is not 1, obviously smallest x can be infinitesimally small 
mathematically, but there's a physical limitation in a computer/hardware.

'''
#importing libraries
import numpy as np

#simple check from the definition above
x = (7.)/3 - (4.)/3 -1
print('The machine accuracy from this simple check is',x)

#b)

#writing a function to calculate the machine accuracy

def machine_acc(func=float):
    
    '''
    Input: a float which is represented in particular number of bits (e.g. 32-bit, 64-bit)
    Output: the machine accuracy/epsilon corresponding to that number.
    Procedure: Initialise the machine accuracy, x as 1. 
               The x is added to 1 and checked if the result isnt equal to 1.
               If it isn't equal to 1, x is divided by 2 and becomes the new x.
               This is run until the result is too small, the result equals to 1. 
               The result is multiplied by 2 to get the original accuracy x 
               because it would have stopped after reaching half of x. 
               
    '''
  
    test_acc = func(1) 
    
    while func(1) + func(test_acc) != func(1):    
      test_acc = func(test_acc) / func(2)
      
    return func(test_acc) * func(2)

#b)check for other formats
    
print('The machine accuracy for a 32 bit (single precision) number is', machine_acc(np.float32)) 
print('The machine accuracy for a 64 bit (double precision) number is', machine_acc(np.float64)) 
print('The machine accuracy for a long double, extended precision number is', machine_acc(np.longdouble)) 

#%%