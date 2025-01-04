# Writing an autograd function from scratch 

# Construct a computational graph of all the variables in the computation

import numpy as np

max_number_of_variables = 10

computational_graph_nodes = {}
gradient_2d_array = np.full((max_number_of_variables, max_number_of_variables), None, dtype=object)


var_counter = 0

class Variable:
    def __init__(self, value, requires_grad=False):
        self.value = value
        self.requires_grad = requires_grad
        if requires_grad:
            global var_counter
            # add the variable to the computational graph
            computational_graph_nodes[self] = var_counter
            var_counter += 1
            

    def __add__(self, other):
        
        if self.requires_grad or other.requires_grad:
            tmp_var = Variable(self.value + other.value, requires_grad=True)
            
            if self.requires_grad:
                gradient_2d_array[computational_graph_nodes[self], computational_graph_nodes[tmp_var]] = 1
            
            if other.requires_grad:
                gradient_2d_array[computational_graph_nodes[other], computational_graph_nodes[tmp_var]] = 1
            
        else:
            tmp_var = Variable(self.value + other.value)

        return tmp_var
            
    
    def __mul__(self, other):
        
        if self.requires_grad or other.requires_grad:
            tmp_var = Variable(self.value * other.value, requires_grad=True) 

            if self.requires_grad:
                gradient_2d_array[computational_graph_nodes[self], computational_graph_nodes[tmp_var]] = other.value
            
            if other.requires_grad:
                gradient_2d_array[computational_graph_nodes[other], computational_graph_nodes[tmp_var]] = self.value
            
        else: 
            tmp_var = Variable(self.value * other.value)
        return tmp_var

    def __sub__(self, other):
        
        if self.requires_grad or other.requires_grad:
            tmp_var = Variable(self.value - other.value, requires_grad=True)

            if self.requires_grad:
                gradient_2d_array[computational_graph_nodes[self], computational_graph_nodes[tmp_var]] = -1
            
            if other.requires_grad:
                gradient_2d_array[computational_graph_nodes[other], computational_graph_nodes[tmp_var]] = -1
            
        else:
            tmp_var = Variable(self.value - other.value)
        
        return tmp_var
    

    def __pow__(self, power):
        if self.requires_grad:
            tmp_var = Variable(self.value**power, requires_grad=True)
            gradient_2d_array[computational_graph_nodes[self], computational_graph_nodes[tmp_var]] = power * self.value**(power-1)
        else:
            tmp_var = Variable(self.value**power)
        return tmp_var


def main():

    

    x1 = Variable(10, requires_grad=True)
    x2 = Variable(2)
    w1 = Variable(3, requires_grad=True)
    w2 = Variable(4, requires_grad=True)
    y_target = Variable(0)

    y = x1**2 * (w1 + x2) * w2
    
    breakpoint()


if __name__ == '__main__':
    main()

