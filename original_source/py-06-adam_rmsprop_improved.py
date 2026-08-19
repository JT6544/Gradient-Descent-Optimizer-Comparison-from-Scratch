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

def gradient_descent(g,alpha_choice,type,max_its,w,beta1=0.9,beta2=0.999,gamma=0.9,eps=1e-8):

    gradient = grad(g)

    # initialisation
    g0 = gradient(w)
    d  = g0                         # first moment (adam): d^0 = grad g(w^0)
    h  = g0 * g0                    # second moment (adam): h^0 = (grad g(w^0))^2
    h_rms = g0 * g0                 # rmsprop accumulator: same pattern for k=0


    weight_history = [w]
    cost_history = [g(w)]

    # main gradient descent loop
    for k in range(1,max_its+1):

        # check if diminishing step rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        grad_eval = gradient(w)

        # take gradient descent step depending on type
        if type == 'normalised':
            grad_eval = grad_eval / (np.sqrt(grad_eval**2) + eps)
            w = w - alpha*grad_eval

        elif type == 'adam':
            d = beta1*d + (1 - beta1)*grad_eval
            h = beta2*h + (1 - beta2)*(grad_eval * grad_eval)
            w = w - alpha * d / (np.sqrt(h) + eps)

        elif type == 'rmsprop':
            h_rms = gamma*h_rms + (1 - gamma)*(grad_eval * grad_eval)
            w = w - alpha * grad_eval / (np.sqrt(h_rms) + eps)

        else:  # plain gradient descent
            w = w - alpha*grad_eval

        # store values
        weight_history.append(w)
        cost_history.append(g(w))

    return weight_history, cost_history


# cost for this example
g = lambda w: w**4
# initial point
w = -1.5; max_its = 200;alpha = 10**(-2);
beta1 = 0.9; beta2 = 0.999; gamma = 0.9;
# run
type=""
weight_history_1,cost_history_1 = gradient_descent(g,alpha,type,max_its,w)

print("weight_history standard")
for i in range(max_its):
    print(i, weight_history_1[i])

# plot the cost function history for a given run
plt.plot(cost_history_1, linestyle='dashed', label="cost, standard")
plt.plot(weight_history_1, label="weight, standard")
plt.xlabel(r'$k$')
plt.legend()
plt.show()


# run
type='normalised'
weight_history_2,cost_history_2 = gradient_descent(g,alpha,type,max_its,w)

print("weight_history normalised")
for i in range(max_its):
    print(i, weight_history_2[i])


print("standard cost, weight:", cost_history_1[max_its], weight_history_1[max_its])
print("normalised cost, weight:", cost_history_2[max_its], weight_history_2[max_its])

# plot the cost function history for a given run
plt.plot(cost_history_2, linestyle='dashed', label="cost, normalised")
plt.plot(weight_history_2, label="weight, normalised")

plt.legend()
plt.show()

# run
type='adam'
weight_history_3,cost_history_3 = gradient_descent(g,alpha,type,max_its,w, beta1, beta2)

print("weight_history normalised")
for i in range(max_its):
    print(i, weight_history_3[i])


print("standard cost, weight:", cost_history_1[max_its], weight_history_1[max_its])
print("normalised cost, weight:", cost_history_2[max_its], weight_history_2[max_its])
print("adam cost, weight:", cost_history_3[max_its], weight_history_3[max_its])

# plot the cost function history for a given run
plt.plot(cost_history_3, linestyle='dashed', label="cost, adam")
plt.plot(weight_history_3, label="weight, adam")

plt.legend()
plt.show()


# run
type='rmsprop'
weight_history_4,cost_history_4 = gradient_descent(g,alpha,type,max_its,w, gamma =0.9)

print("weight_history normalised")
for i in range(max_its):
    print(i, weight_history_4[i])


print("standard cost, weight:", cost_history_1[max_its], weight_history_1[max_its])
print("normalised cost, weight:", cost_history_2[max_its], weight_history_2[max_its])
print("adam cost, weight:", cost_history_3[max_its], weight_history_3[max_its])
print("rmsprop cost, weight:", cost_history_4[max_its], weight_history_4[max_its])

# plot the cost function history for a given run
plt.plot(cost_history_4, linestyle='dashed', label="cost, rmsprop")
plt.plot(weight_history_4, label="weight, rmsprop")

plt.legend()
plt.show()
