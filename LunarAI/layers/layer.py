try:
    from ai_libs.array_type import Array
    from activations import *
except:
    from LunarAI.ai_libs.array_type import Array
    from LunarAI.activations.activations import *

class Layer:
    def __init__(self , size = 20 , activation = relu , inputs = 20) -> None:
        self.W = Array( [ [0]*inputs ]*size )
        self.B = Array( [[0]*size] )
        self.a = activation
        return None
    def __call__(self, X) -> Array:
        if not isinstance(X , Array):
            raise TypeError
        Z = X.dot(self.W.T) + self.B
        P = self.a(Z)
        return P
