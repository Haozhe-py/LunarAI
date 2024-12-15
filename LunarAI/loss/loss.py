import LunarAI.loss.functions as f
import LunarAI.loss.deda as d

class Loss:
    def __init__(self , function , deda) -> None:
        self.function = function
        self.deda = deda
        return None
    def __call__(self,y,p):
        return self.function(y,p)
    def dadz(self,p):
        return self.deda(p)

mean_squared_error = Loss(f.mean_squared_error , d.mean_squared_error)
