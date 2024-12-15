#SGD
from LunarAI.loss.loss import *

class SGD:
    def __init__(self, learning_rate=0.01, loss_function = mean_squared_error) -> None:
        self.learning_rate = learning_rate
        self.loss = loss_function
        return None
    def __update(self, params, grid) -> None:
        for key in params.keys():
            params[key] -= ( self.learning_rate * grid[key] )
            continue
        return None
    def __deda(self , model , y_train, P):   #calculate de/da, e=loss function, a=activation function
        deda = [0]*len(model.net)
        dedz = [0]*len(model.net)
        for ii in range(1 , len(model.net)+1):
            i =  -ii
            if i == -1:
                deda[i] = self.loss.deda(y_train,P)
                dadz = model.net[i].a.dadz(model.net[i].Z)
                dedz[i] = deda[i] * dadz
            else:
                deda[i] = dedz[i+1].dot(model.net[i+1].params['W'].T)
                dadz = model.net[i].a.dadz(model.net[i].Z)
                dedz[i] = deda[i] * dadz
            continue
        return deda

    def __call__(self, x_train, y_train, model):
        P = model(x_train)
        L = self.loss(y_train,P)
        print(f'loss={abs(P-y_train)}')
        params , grid = {} , {}
        deda_ = self.__deda(model,y_train,P)
        for ii in range(1 , len(model.net)+1):
            i = ii * (-1)
            deda = ai.ai_libs.array_type.Array([deda_[i]])
            dadz = model.net[i].a.dadz(model.net[i].Z)
            for neuron in range(model.net[i].size):
                dedz = deda[0][neuron]*dadz[0][neuron]
                for p in model.net[i].params.keys():
                    dzdp = model.net[i].d[p]()
                    dedp = dedz * dzdp
                    params[p] = model.net[i].params[p]
                    grid[p] = dedp
                    self.__update(params,grid)
                    continue
                continue
            continue
        return None