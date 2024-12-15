
class Model:
    def __init__(self):
        self.net = []
        return
    def add(self,layer):
        self.net.append(layer)
        return
    def __call__(self,x):
        for i in range(len(self.net)):
            x=self.net[i](x)
            continue
        return x

