'''
@author: Dani Atalla
Data: 09/09/21
Time:15:30
'''

import numpy as np
import matplotlib.pyplot as plt

# define a loss function 
def loss_function(x,y,a,b,c):
# our predicted model is pred_y    
    pred_y = a*(x**2)+b*np.log(x)+c
# the loss function is the MSE loss function
    loss_func=(((pred_y-y)**2).mean())*0.5
    return (pred_y,loss_func)

# define the gradient calculater for the loss function
def gradient_calculater(pred_y,y,x):
#  we calculate the gradient of : (((pred_y-y)**2).mean())*0.5 when pred_y = a*(x**2)+b*np.log(x)+c
# the partial derivatives of the loss function by it's paramaters are: 
        der_a=( ((pred_y-y)*(x**2)).mean() )
        der_b=( ((pred_y-y)*np.log(x) ).mean() )
        der_c=( ((pred_y-y)).mean() )
        # and the gradient itsel is returned
        return (der_a,der_b,der_c)
    
    

# define the gradient descent function alogorithm:
def gradient_alog(x,y,a1,b1,c1,L,iterations):
# using the copies of the parameters so we can use the function for differnet learning rates   
    a=a1.copy()
    b=b1.copy()
    c=c1.copy()
# adding 1 to number of iteration so range will return a list of numbers from 0 to iterations (included)
    for i in range((iterations+1)):
        # calculate the loss function for the parameters and store the result in pred_y and loss
        (pred_y,loss)=loss_function(x,y,a,b,c)
        # calculate the gradient for each prediction and data
        (der_a,der_b,der_c) =gradient_calculater(pred_y,y,x)
        # update the paramteres with the gradient multiply by L (the learning rate) 
        a=a-L*der_a
        b=b-L*der_b
        c=c-L*der_c
    # after all iteration are done, return the best paramteres found    
    return (a,b,c)


# the data we have from the question x values and y values:
    
x = np.array([0.001,1,2])
y = np.array([1,3,7])
"""
our model is :
predicted_y = a*(x**2)+b*log(x)+c
so we need to set initial values of a,b,c as:
a=2
b=2
c=0
"""
# initial values of parameters
# we will define the parameters as float so we won't loss any data
starting_point = np.array([2.0,2.0,0.0],dtype=np.float64)

# for the learning rate L=0.1, and number of iteration = 100
iterations = 100
L=0.1
print("\n the loss function value at the starting point (2,2,0) after "+str(iterations)+" iterations "+",using"+" L= "+str(L)+" is : ")
(a,b,c)=gradient_alog(x,y,starting_point[0],starting_point[1],starting_point[2],L,iterations)
print("\n"+str(loss_function(x, y, a, b, c)[1])+"\n")
L=1
print("\n the loss function value at the starting point (2,2,0) after "+str(iterations)+" iterations "+",using"+" L= "+str(L)+" is : ")
(a,b,c)=gradient_alog(x,y,starting_point[0],starting_point[1],starting_point[2],L,iterations)
print("\n"+str(loss_function(x, y, a, b, c)[1])+"\n")




