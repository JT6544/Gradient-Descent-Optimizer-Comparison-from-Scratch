# import basic libraries and autograd wrapped numpy
import sys
sys.path.append('../')
import autograd.numpy as np
import copy
import matplotlib.pyplot as plt

from autograd import grad

# this is needed to compensate for matplotlib notebook's tendancy to blow up images when plotted inline
#matplotlib notebook
from matplotlib import rcParams
rcParams['figure.autolayout'] = True
plt.rcParams['text.usetex'] = True

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,norm,max_its,w):
    
    gradient=grad(g)
    
    # run the gradient descent loop
    weight_history = [w] # container for weight history
    cost_history = [g(w)] # container for corresponding cost function history

    for k in range(max_its):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k+1)
        else:
            alpha = alpha_choice

        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)
        
        # take gradient descent step
        if norm == 'normalised':
            grad_eval = np.sign(gradient(w))
        else:
            grad_eval = gradient(w)
        w = w - alpha*grad_eval
        # collect final weights
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history
 

# cost for this example
g = lambda w: w**4
# initial point
w = -1.5; max_its = 200;alpha = 10**(-2);

# run
norm=1
weight_history_1,cost_history_1 = gradient_descent(g,alpha,norm,max_its,w)

print("standard cost, weight:", cost_history_1[max_its], weight_history_1[max_its])

# plot the cost function history for a given run
plt.plot(cost_history_1, linestyle='dashed', label="cost, standard")
plt.plot(weight_history_1, label="weight, standard")


# run
norm='normalised'
weight_history_2,cost_history_2 = gradient_descent(g,alpha,norm,max_its,w)

print("normalised cost, weight:", cost_history_2[max_its], weight_history_2[max_its])

# plot the cost function history for a given run
plt.xlabel(r'$k$')
plt.plot(cost_history_2, linestyle='dashed', label="cost, normalised")
plt.plot(weight_history_2, label="weight, normalised")

plt.legend()
plt.show()
