from LunarAI.ai_libs.array_type import Array
from LunarAI.activations.activations import Activation

class Embedding:
    def __init__(self, word_vec_size, word_vec_d) -> None:
        W = Array(
            [[0]*word_vec_d]*word_vec_size
        )
        self.X, self.Z = None, None
        self.params = {'W':W}
        self.d = {'W':(lambda:self.X)}
        self.a = Activation(lambda z:z , lambda z:1)
        return None
    def __call__(self, X) -> Array:
        if not isinstance(X, Array):
            raise TypeError
        return X.dot(self.params['W'])