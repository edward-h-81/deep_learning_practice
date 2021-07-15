import numpy as np

# video 6 in Valerio's deep learning for audio series

class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1]) # creating a matrix
            self.weights.append(w)

    def forward_propogate(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w) # performs a matrix multiplication

            # calculate the activations
            activations = self.sigmoid(net_inputs)

        return activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

    # create an MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward prop
    outputs = mlp.forward_propogate(inputs)

    # print results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))

# below is my own function - its purpose is to show the inner workings of the layers and weights with debug

# def mlp_test(num_inputs=3, num_hidden=[3,5], num_outputs=2):
#
#     layers = [num_inputs] + num_hidden + [num_outputs]
#
#     weights = []
#     for i in range(len(layers)-1):
#         w = np.random.rand(layers[i], layers[i+1]) # creating a matrix
#         weights.append(w)
#
#     return weights
#
# var = mlp_test()
# print(var)