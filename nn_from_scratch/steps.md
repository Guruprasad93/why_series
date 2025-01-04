# Detailing the steps for training a simple neural network from scratch

### Objective: Training a 2 layered neural network to become an AND-gate

AND-Gate Truth table: 
(x1, x2) => y_target

For example: 
(0,0) -> 0
(0,1) -> 0
(1,0) -> 0
(1,1) -> 1


### Define the architecture of the neural network

[Insert Image]

Defining a weight matrix (to solve for)
w = [w_1, w_2]

### neural network Inference (prediction step)

def sigmoid(a: float): 
    # write the function of sigmoid
    sigmoid_output = exp()
    return sigmoid_output
    

def forward(inputs: (x1, x2), weights: (w1, w2)):

    output_nn = sigmoid(weights[0] * x1 + weights[1] * x2)
    return output_nn # This is a single value (y)


### Loss function to be optmized for: 

def loss(y_target, y: output of the network for given inputs):

    loss_value = (y_target - y)**2
    return loss_value


### Gradient of the loss function wrt the weights 

def loss_gradient_wrt_wts(y, inputs: (x1, x2), y_target):
    
    y_grad_wrt_w1 = sigmoid(y)/(1+exp(y)) * x1
    y_grad_wrt_w2 = sigmoid(y)/(1+exp(y)) * x2


    loss_grad_wrt_w1 = 2*(y-y_t)*y_grad_wrt_w1
    loss_grad_wrt_w2 = 2*(y-y_t)*y_grad_wrt_w2
    
    return (loss_grad_wrt_w1, loss_grad_wrt_w2)
    

### pull examples from the dataset (truth table) one after the other

def main():

    w = initialize_weights_of_neural_network()
    learning_rate(eta) = 0.01

    loss_value_store = []

    for epoch in range(10):

        for entry in truth_table: 

            x1, x2, y_target = entry
            y = forward((x1,x2), w)
            loss_value = loss(y_target, y)

            # find slope (gradient) of loss wrt the different weights in the network
            # Grad(L) wrt w1, Grad(L) wrt w2

            # TODO: write our own autograd function 

            loss_grad_wrt_w1, loss_grad_wrt_w2 = loss_gradient_wrt_wts(y, (x1, x2), y_target)

            # Find the new weights (optimization)

            w1_new = w1 - eta * loss_grad_wrt_w1
            w2_new = w2 - eta * loss_grad_wrt_w2

            w1 = w1_new
            w2 = w2_new
        
        loss_value_store.append(loss_value)

    

    



        




    