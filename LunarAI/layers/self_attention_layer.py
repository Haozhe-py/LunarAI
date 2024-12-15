from LunarAI.ai_libs.array_type import Array
from LunarAI.activations.activations import Activation, softmax
from math import sqrt

class SelfAttentionLayer:
    def __init__(self, input_d, input_size) -> None:
        WQ = Array(
            [[0]*input_d]*input_d
        )
        WK = Array(
            [[0]*input_d]*input_d
        )
        WV = Array(
            [[0]*input_d]*input_d
        )
        self.dk_sqrt = sqrt(input_d)
        self.params = {'WQ':WQ,'WK':WK,'WV':WV}
        self.X,self.Z = None,None
        self.d = {'WQ':(lambda:self.X),'WK':(lambda:self.X),'WV':(lambda:self.X)} #
        self.a = Activation(lambda z:z , lambda z:1)
        return None
    def __call__(self, XQ, XK, XV) -> Array:
        Q, K, V = XQ.dot(self.params['WQ']),XK.dot(self.params['WK']),XV.dot(self.params['WV'])
        score = Q.dot(K.T)
        score /= self.dk_sqrt
        score = softmax(score)
        i_s, j_s = score.size()
        Z = Array([[0]*i_s])
        for i in range(i_s):
            for j in range(j_s):
                Z[0][i] += score[i][j]*(Array([V[j]]))
                continue
            continue
        self.Z = Z
        return Z