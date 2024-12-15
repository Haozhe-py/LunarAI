from LunarAI.ai_libs.array_type import Array
from LunarAI.activations.activations import *

class Layer:
    def __init__(self , size = 20 , activation = relu , inputs = 20) -> None:
        W = Array( [ [0]*inputs ]*size )
        B = Array( [[0]*size] )
        self.size, self.inputs = size, inputs
        self.a = activation
        self.Z, self.X = None, None
        self.params = {'W':W , 'B':B}
        self.d = {'W':lambda:self.X,'B':lambda:1}    #dz/dw & dz/db
        return None
    def __call__(self, X) -> Array:
        if not isinstance(X , Array):
            raise TypeError
        self.X = X
        Z = X.dot(self.params['W'].T) + self.params['B']
        P = self.a(Z)
        self.Z = Z
        return P
