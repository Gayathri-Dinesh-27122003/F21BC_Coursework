import numpy as np

class ActivationFunctions:

    #Class that contains various activation functions

    @staticmethod
    def relu(x):
        "Rectified Linear Unit = max(0, x)"
        return np.maximum(0,x)
    
    @staticmethod
    def sigmoid(x):
        "Sigmoid = 1 / (1 + exp(-x))"
        x = np.clip(x, -500,500)
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def linear(x):
        "Linear = x"
        return x
    
    @staticmethod
    def softmax(x):
        "Softmax = exp(x) / sum(exp(x))"
        exp_x = np.exp(x-np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims = True)
    
    @staticmethod
    def tanh(x):
        "tanh = (exp(x))-(exp(-x)) / (exp(x) + exp(-x))"
        x = np.clip(x, -500,500)
        return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        "Leaky Relu = max(alpha*x, x)"
        return np.maximum(alpha*x, x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        "ELU (Exponential Linear Unit) = x if x > 0 else alpha * (exp(x) - 1)"
        x = np.clip(x, -500, 500)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def selu(x):
        "SELU (Scaled ELU) with alpha=1.6733 and scale=1.0507"
        alpha = 1.6733
        scale = 1.0507
        x = np.clip(x, -500, 500)
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def get_function(name):
        #Returns the activation function based on the name provided
        activations = {
            'relu': ActivationFunctions.relu,
            'sigmoid': ActivationFunctions.sigmoid,
            'linear': ActivationFunctions.linear,
            'softmax': ActivationFunctions.softmax,
            'tanh': ActivationFunctions.tanh,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'elu': ActivationFunctions.elu,
            'selu': ActivationFunctions.selu
            }
        if name not in activations:
            raise ValueError(f"Activation function '{name}' is not supported.")
        return activations[name]
    

class NetworkArchitecture:

    #Class to define network architecture

    def __init__(self, layer_sizes, activation_functions):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self._validate_architecture()

    def _validate_architecture(self):
        "Validate the provided architecture"
        if len(self.layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers.")
        
        if len(self.activation_functions) != len(self.layer_sizes) -1:
            raise ValueError(f"Number of activation functions ({len(self.activation_functions)})"
                             f"must be equal to number of layers - 1 ({len(self.layer_sizes) - 1})")
        
        #Check for valid activation function names
        valid_activations = ['relu', 'sigmoid', 'linear', 'softmax', 'tanh', 'leaky_relu', 'elu', 'selu']

        for activation in self.activation_functions:
            if activation not in valid_activations:
                raise ValueError(f"Invalid activation function: {activation}. Valid options: {valid_activations}")
            
    def get_layer_info(self, layer_index):
        #Get information about the layer
        if layer_index >= len(self.layer_sizes):
            raise ValueError(f"Layer index {layer_index} out of range.")
        
        info = {
            'layer_number': layer_index +1,
            'neurons': self.layer_sizes[layer_index],
            'layer_type': 'input' if layer_index ==0 else
                          'output' if layer_index == len(self.layer_sizes) -1 else 'hidden'}
        
        
        if layer_index<len(self.activation_functions):
            info['activation'] = self.activation_functions[layer_index]
        
        return info
    
    def get_network_summary(self):
        #Get complete summary of the network architecture"

        summary = {
            'total_layers': len(self.layer_sizes),
            'input_neurons': self.layer_sizes[0],
            'output_neurons': self.layer_sizes[-1],
            'hidden_layers': len(self.layer_sizes) -2,
            'total_neurons': sum(self.layer_sizes),
            'activation_functions': self.activation_functions,
            'layer_sizes': self.layer_sizes

        }
        return summary

class NeuralNetwork:

    #Configurable neural network for regression tasks

    def __init__(self, layer_sizes, activation_functions):
        

        self.architecture = NetworkArchitecture(layer_sizes, activation_functions)
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        #Initialize weights and biases for all layers
        self.weights = []
        self.biases =[]

        for i in range(len(self.architecture.layer_sizes) - 1):
            input_size = self.architecture.layer_sizes[i]
            output_size = self.architecture.layer_sizes[i+1]
            activation = self.architecture.activation_functions[i]

            #Initialisation for relu and leaky relu
            if activation in ['relu', 'leaky_relu']:
                scale = np.sqrt(2.0/input_size)
            else:
                scale = np.sqrt(1.0/input_size)

            #Initialize weights and baises
            weight_matrix = np.random.randn(input_size, output_size) * scale
            bias_vector = np.zeros((1,output_size))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, X):
       
        current_output = X
        
        for i in range(len(self.weights)):
            # Linear transformation: Z = XW + b
            z = np.dot(current_output, self.weights[i]) + self.biases[i]
            
            # Apply activation function
            activation_func = ActivationFunctions.get_function(
                self.architecture.activation_functions[i]
            )
            current_output = activation_func(z)
        
        return current_output
    
    def predict(self, X):

        return self.forward(X)
    
    # PSO integration methods
    def set_parameters(self, parameter_vector):
        
        pointer = 0
        
        for i in range(len(self.weights)):
            # Extract and reshape weights
            weight_size = self.weights[i].size
            self.weights[i] = parameter_vector[pointer:pointer + weight_size].reshape(self.weights[i].shape)
            pointer += weight_size
            
            # Extract and reshape biases
            bias_size = self.biases[i].size
            self.biases[i] = parameter_vector[pointer:pointer + bias_size].reshape(self.biases[i].shape)
            pointer += bias_size
    
    def get_parameters(self):
        
        parameter_list = []
        
        for i in range(len(self.weights)):
            parameter_list.append(self.weights[i].flatten())
            parameter_list.append(self.biases[i].flatten())
        
        return np.concatenate(parameter_list)
    
    def get_parameter_count(self):
       
        total_parameters = 0
        
        for i in range(len(self.weights)):
            total_parameters += self.weights[i].size + self.biases[i].size
        
        return total_parameters
    
    def get_network_info(self):
       
        arch_info = self.architecture.get_network_summary()
        arch_info['total_parameters'] = self.get_parameter_count()
        return arch_info