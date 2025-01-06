## Script to train a simple NN from scratch to mimic an AND gate

import random
from math import exp

TRUTH_TABLE = [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]


def initialize_weights_of_neural_network():
    w1 = random.random()
    w2 = random.random()
    return [w1, w2]

# the sigmoid_like function rises from x=0 to x=1
def sigmoid_like(x):
    if x <0.4: 
        return 0
    if x >0.6: 
        return 1
    return 5*x-2

def sigmoid(x: float): 
    # write the function of sigmoid
    sigmoid_output = exp(x)/(1 + exp(x))
    return sigmoid_output
    
def forward(inputs: tuple[float, float], weights: tuple[float, float], fn) -> float: 

    x1, x2 = inputs
    y = fn(weights[0] * x1 + weights[1] * x2)
    return y # This is a single value (y) that the network predicts

def loss(y_target, y: float) -> float:

    loss_value = (y_target - y)**2
    return loss_value


def loss_gradient_wrt_wts(y: float, inputs: tuple[float, float], y_target: float, fn) -> tuple[float, float]:
    
    x1, x2 = inputs
    
    if fn == sigmoid:
        y_grad_wrt_w1 = sigmoid(y)/(1+exp(y)) * x1
        y_grad_wrt_w2 = sigmoid(y)/(1+exp(y)) * x2

    elif fn == sigmoid_like:
        if y < 0.4: 
            y_grad_wrt_w1 = 0
            y_grad_wrt_w2 = 0
        elif y > 0.6:
            y_grad_wrt_w1 = 0
            y_grad_wrt_w2 = 0
        else:
            y_grad_wrt_w1 = 5 * x1
            y_grad_wrt_w2 = 5 * x2

    loss_grad_wrt_w1 = 2*(y-y_target)*y_grad_wrt_w1
    loss_grad_wrt_w2 = 2*(y-y_target)*y_grad_wrt_w2
    
    return (loss_grad_wrt_w1, loss_grad_wrt_w2)

def main():

    w = initialize_weights_of_neural_network()
    learning_rate = 0.01

    loss_value_store = []
    non_linear_function = sigmoid_like

    for epoch in range(10):
        
        loss_value_per_epoch = 0
        
        loss_grad_wrt_w1_total = 0
        loss_grad_wrt_w2_total = 0
        
        for entry in TRUTH_TABLE: 

            x1, x2, y_target = entry
            y = forward((x1,x2), w, fn = non_linear_function)
            loss_value = loss(y_target, y)

            # find slope (gradient) of loss wrt the different weights in the network
            # Grad(L) wrt w1, Grad(L) wrt w2

            # TODO: write our own autograd function 

            loss_grad_wrt_w1, loss_grad_wrt_w2 = loss_gradient_wrt_wts(y, (x1, x2), y_target, fn = non_linear_function)

            loss_grad_wrt_w1_total += loss_grad_wrt_w1
            loss_grad_wrt_w2_total += loss_grad_wrt_w2

            loss_value_per_epoch += loss_value

        # Find the new weights (optimization)
        breakpoint()
        
        w1_new = w[0] - learning_rate * loss_grad_wrt_w1_total
        w2_new = w[1] - learning_rate * loss_grad_wrt_w2_total

        w[0] = w1_new
        w[1] = w2_new

        loss_value_store.append(loss_value_per_epoch)
        print("Loss value after epoch ", epoch, " is ", loss_value_per_epoch)

        
if __name__ == "__main__":
    main()