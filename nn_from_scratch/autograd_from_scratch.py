# Writing an autograd function from scratch 

import numpy as np
import random
import pickle

MAX_VARIABLES = 10 # Includes all intermediate variables within the computation graph

#  Set up compuational graph nodes and the gradients of all edges
computational_graph_nodes = {}
gradient_2d_array = np.full((MAX_VARIABLES, MAX_VARIABLES), None, dtype=object)

# variable counter
var_counter = 0

class Variable:
    def __init__(self, value, requires_grad=False, grad=0):
        self.value = value
        self.grad = grad
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
                gradient_2d_array[computational_graph_nodes[self], computational_graph_nodes[tmp_var]] = 1
            
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


    def backward(self):
        # find the path from self node to all nodes in the computational graph
        global computational_graph_nodes, gradient_2d_array
        
        gradient_2d_array_edit = gradient_2d_array[:len(computational_graph_nodes), :len(computational_graph_nodes)]
        adj_matrix = np.where(gradient_2d_array_edit != None, 1, 0)
        
        start_node = computational_graph_nodes[self]
        # end_nodes - list of all nodes except the start node
        end_nodes = list(computational_graph_nodes.values())
        end_nodes.remove(start_node)

        # find all the paths until terminal (leaf) node
        paths = []
        queue = [] 
        # insert the start node into a queue 
        queue.append([start_node])
        
        while True: # until leaf node hasn't been reached
            
            # do a DFS style traversal 
            # pop the last element from the queue
            node_path = queue.pop()
            connected_nodes, start_is_leaf_node = explore_previous_nodes(adj_matrix, node_path[-1])
            
            for ii in connected_nodes:
                tmp = node_path.copy()
                tmp.append(ii)
                queue.append(tmp)
                
            if start_is_leaf_node:
                paths.append(node_path)
            
            if len(queue) == 0:
                break

        # find the gradient of the self variable wrt all the leaf nodes
        for path in paths: 
            gradient = 1
            for ii in range(len(path)-1):
                gradient = gradient * gradient_2d_array_edit[path[ii+1], path[ii]]
                key_node = [k for k, v in computational_graph_nodes.items() if v == path[ii+1]][0]
                
                if key_node.grad == None:
                    key_node.grad = gradient   
                else: 
                    key_node.grad += gradient
            
            # print("Gradient of ", path[0], " wrt ", path[-1], " is ", gradient)
            # computational_graph_nodes[path[-1]].grad = gradient
        leaf_nodes = {}
        for path in paths: 
            for ii in range(len(path)-1):
                key_node = [k for k, v in computational_graph_nodes.items() if v == path[ii+1]][0]
                if path[ii + 1] == path[-1]:
                    leaf_nodes[key_node] = path[ii+1]
                    # print("Gradient of ", path[0], " wrt ", path[ii+1], " (leaf node) is ", key_node.grad)
                else:
                    pass
                    # print("Gradient of ", path[0], " wrt ", path[ii+1], " is ", key_node.grad)
                # print("Gradient of ", path[0], " wrt ", path[-1], " is ", computational_graph_nodes[path[-1]].grad)
        
        ###  delete all nodes except the leaf nodes 
        
        # Collect keys to delete in a list
        keys_to_delete = [key_node for key_node in computational_graph_nodes if key_node not in leaf_nodes]

        # Delete the collected keys from the dictionary
        for key_node in keys_to_delete:
            del computational_graph_nodes[key_node]
        
        # delete all the gradients except the leaf nodes and set to None
        for ii in range(len(gradient_2d_array)):
            if ii not in leaf_nodes.values():
                gradient_2d_array[:, ii] = None
                gradient_2d_array[ii, :] = None

        global var_counter
        var_counter = len(leaf_nodes)
        return 
        
        
def explore_previous_nodes(adj_matrix, start_node): 
    start_is_leaf_node = True
    connected_nodes = []
    for ii in range(len(adj_matrix)):
        if adj_matrix[ii, start_node] == 1:
            connected_nodes.append(ii)
            start_is_leaf_node = False

    return connected_nodes, start_is_leaf_node




def main():

    
    # x1 = Variable(0.5)
    # x2 = Variable(0.1)
    # w1 = Variable(0.01, requires_grad=True)
    # w2 = Variable(0.4, requires_grad=True)
    # y_target = Variable(1)

    # y = x1**2 * (w1 + x2) * w2
    # L = (y - y_target)**2
    # L.backward()

    # Find square root of z=103

    # pick a random value of x
    init_value = random.choice(range(30, 40))
    print("Initial value of x is ", init_value)
    
    x = Variable(init_value, requires_grad=True)
    z = Variable(45)

    learning_rate = 1e-9 # TODO: Change to adaptive learning rate
    iter = 0

    loss_values = []
    
    while True:
        
        y = x**2
        L = (y - z)**2

        L.backward()

        print("Value of x after iteration ", iter, " is ", x.value)
        print("Value of y after iteration ", iter, " is ", y.value)
        print("Value of L after iteration ", iter, " is ", L.value)

        x.value = x.value - learning_rate * x.grad
        loss_values.append(L.value)
        iter += 1

        pickle.dump(loss_values, open("nn_from_scratch/sq_fn_loss.pkl", "wb"))
        
        if abs(L.value) < 0.01: # Termination condition
            break


if __name__ == '__main__':
    main()

