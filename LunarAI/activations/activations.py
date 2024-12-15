try:
    import activations.functions as f
    import activations.dadz as d
except ImportError:
    import LunarAI.activations.functions as f
    import LunarAI.activations.dadz as d


class Activation:
    def __init__(self , function , dadz) -> None:
        self.function = function
        self._dadz = dadz
        return None
    def __call__(self,z):
        return self.function(z)
    def dadz(self,z):
        return self._dadz(z)

sigmoid = Activation(f.sigmoid , d.sigmoid)
relu = Activation(f.relu , d.relu)

