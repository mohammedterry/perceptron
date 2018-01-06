import numpy as np 
import random 

class Synapse:  
    "a uni-directional link between two neurons.  Neuron_1 ->-Synapse->- Neuron_2"
    w = random.random()
    i = 0
    o = 0
    def predict(self, i):
        self.i = i
        self.o = self.i * self.w
        return self.o
    def update(self, expected):
        "compare predicted to expected output, then adjust weight to improve future predictions"
        d =  (expected - self.o) * self.f(self.o)
        self.w = self.i * d
        return d
    def f(self, x,deriv=False):
        "non-linear input-output mapping function"
        if deriv:
            return x*(1-x)
        return 1/(1+np.exp(-x))

def train(input, output):
    l0 = Synapse()
    for i in range(100):
        l0.predict(input)
        print(l0.update(output))  
    return l0

trained_neuron = train(1, 10)
print("trained neuron")
print(trained_neuron.predict(1))
