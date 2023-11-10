import numpy as np
from affine_function import Affine
from sigmoid_function import Sigmoid

class ArtificialNeuron:
    
    def __init__(self, w, b):
        self.affine_neuron = Affine(w=w, b=b)
        self.activation_neuron = Sigmoid()
        
    def __call__(self, x) -> np.ndarray:
        return self.activation_neuron(self.affine_neuron(x))