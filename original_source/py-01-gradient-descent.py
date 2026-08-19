# import basic libraries and autograd wrapped numpy
import sys
sys.path.append('../')
#import autograd.numpy as np
import copy
import matplotlib.pyplot as plt

# this is needed to compensate for matplotlib notebook's tendancy to blow up images when plotted inline
#matplotlib notebook
from matplotlib import rcParams
rcParams['figure.autolayout'] = True

# gradient descent function - inputs: alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(alpha,max_its,w):
    # cost for this example
    g = lambda w: 1/50*(w**4 + w**2 + 10*w)
    
    # the gradient function for this example
    gradient = lambda w: 1/50*(4*w**3 + 2*w + 10)
    
    
    # run the gradient descent loop
    cost_history = [g(w)] # container for corresponding cost function history
    weight_history = [w] # container for weight history
    
    for k in range(1,max_its+1):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
        
        # take gradient descent step
        w = w - alpha*grad_eval
        # collect final weights
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history

# initial point
w = 2.0
max_its = 1000

# runs
alpha = 10**(0)
weight_history_1, cost_history_1 = gradient_descent(alpha,max_its,w)

alpha = 10**(-1)
weight_history_2, cost_history_2 = gradient_descent(alpha,max_its,w)

alpha = 10**(-2)
weight_history_3, cost_history_3 = gradient_descent(alpha,max_its,w)

print("value of g(w) at minimum is g(w_min)=-0.167")
# plot the cost function history for a given run
plt.plot(cost_history_1, label='alpha=1')
plt.plot(cost_history_2, label='alpha=0.1')
plt.plot(cost_history_3, label='alpha=0.01')
plt.axhline(y=-0.167, linestyle='--')
plt.xlabel('time')
plt.ylabel('value of g(w)')

plt.legend()
plt.show()

print("minimum is at w=-1.234")
# plot the weight history for a given run
plt.plot(weight_history_1, label='alpha=1')
plt.plot(weight_history_2, label='alpha=0.1')
plt.plot(weight_history_3, label='alpha=0.01')
plt.axhline(y=-1.234, linestyle='--')
plt.xlabel('time')
plt.ylabel('w')

plt.legend()
plt.show()
