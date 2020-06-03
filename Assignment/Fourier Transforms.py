'''
COMPUTATIONAL PHYSICS - ASSIGNMENT 
QUESTION 3 : FOURIER TRANSFORMS
'''

#importing appropriate libraries
import numpy as np
import matplotlib.pyplot as plt


'''
We are given a signal function, h(t) and a response function, g(t).
We are required to find the convolution (h * g)(t). 
We can use convolution theorem.
So, we find the fourier transforms of h(t) and g(t) and mulitply them together.
We then take the inverse fourier transform of the product.
The result is the convolution (h * g)(t) we want. 
We are interested in the real part of Fourier Transforms as well as of inverse 
Fourier Transforms.

We did not consider the normalisation factor which should be multiplied with 
the convolved signal. 

'''

#defining signal function, h(t)
def h(t):
    
   '''
   Input: the time value
   Output: the value of h at that time value
   Procedure: Gives the value of function h for a given time value, t.
   '''
    
   if 3 <= t <= 5:
     h = 4
   else:
     h = 0
   return h 

h = np.vectorize(h)            #vectorising to avoid boolean comparison errors


#defining response function -- gaussian
def g(t):
   '''
   Input: the time value, t
   Output: the value of g at that time value
   Procedure: Gives the value of function g which is a gaussian for a given 
   time value, t.
   '''
   return (1/np.sqrt(2*np.pi)) * np.exp(-1*(t**2)/2)


#preparing the time domain:
N = 1024                #number of samples ---> N = 2**10 , so no padding
T = 20                  #period (-10, 10)
dt = T/N                #sample rate/smallest time interval between samples

t = np.linspace(-T/2,T/2, N)         #defining the time range

#preparing the frequency domain:
w_min = 2*np.pi/T                   #minimum frequency
w_max = np.pi/dt                    #Nyquist/max frequency


'''
As we will see in the plots below, the standard FT gives the aliased output, 
resulting in two spikes on opposite sides of the plot. 
To rectify this, we use np.fft.fftshift which shifts the zero-frequency 
component to the center of the spectrum.
This gives us the more familiar plots.
We need to use two differrent sets of x-range for frequencies.
One is (-w_max, w_max)
One is (0, 2*w_max) --> used by standard FT (aliased)

'''
#frequency range 1 (-w_max, w_max) ---> to be used in shifted fourier plot
freq = []
for i in range(-N//2,N//2):
  w_p = i*2*np.pi/(N*dt)
  freq.append(w_p)

#frequncy range 2 (0, 2*w_max)   ---> to be used in unshifted fourier plot
freq_2 = []
for j in range(0, N):
  w_p_2 = j*2*np.pi/(N*dt)
  freq_2.append(w_p_2)


#plotting h(t)
plt.figure(1)                                 #to be named figure #1
plt.title('Signal function h(t)')
plt.xlabel('Time, t')
plt.ylabel('h(t)')
plt.plot(t, h(t), color = 'red')
plt.grid()
plt.show()

#plotting g(t)
plt.figure(2)
plt.title('Response function g(t)')
plt.xlabel('Time, t')
plt.ylabel('g(t)')
plt.plot(t, g(t), color = 'blue')
plt.grid()
plt.show()


#plotting the FT of h
F_h = np.fft.fft(h(t)) *dt                  #fourier of h, multipled by dt 
plt.figure(3)
plt.title('Fourier of h(t), H(w)')
plt.xlabel('Frequency, w')
plt.ylabel('Amplitude')
plt.plot(freq_2, abs(F_h), color = 'red')   #plot of fourier, using the appropriate x-range 
plt.grid()
plt.show()

#plotting the shifted FT of h
F_h_2 = np.fft.fftshift(F_h)                #shifting the fourier of h to be centred at zero
plt.figure(4)
plt.title('Shifted fourier of h(t), H(w)')
plt.xlabel('Frequency, w')
plt.ylabel('Amplitude')
plt.plot(freq, abs(F_h_2), color = 'red')   #plot of shifted fourier, using the appropriate x_range
plt.grid()
plt.show()


#plotting the FT of g 
F_g = np.fft.fft(g(t)) *dt                   #fourier of g, multipled by dt
plt.figure(5)
plt.title('Fourier of g(t), G(w)')
plt.xlabel('Frequency, w')
plt.ylabel('Amplitude')
plt.plot(freq_2, abs(F_g), color = 'blue')   #plot of fourier, using the appropriate x-range
plt.grid()
plt.show()

#plotting the shifted FT of g
F_g_2 = np.fft.fftshift(F_g)                  #shifting the fourier of g to be centred at zero
plt.figure(6)
plt.title('Shifted fourier of g(t), G(w)')
plt.xlabel('Frequency, w')
plt.ylabel('Amplitude')
plt.plot(freq, abs(F_g_2), color = 'blue' )   #plot of shifted fourier, using the appropriate x_range   
plt.grid()
plt.show()


#plotting the convoluted signal
C_gh = np.fft.fft(h(t)) * np.fft.fft(g(t))    #product of fourier
c_gh = np.fft.ifft(C_gh)  * dt                #inverse fourier of product, multipled by dt
c_gh_2 = np.fft.ifftshift(c_gh)               #shifting the inverse fourier to be centred at zero
plt.figure(7)
plt.title('Convolution (h * g)(t)')
plt.xlabel('Time, t')
plt.ylabel('Amplitude')
plt.plot(t, abs(c_gh_2), color = 'purple')     #only taking the absolute value of the convoluted signal
plt.grid()
plt.show()

#%%