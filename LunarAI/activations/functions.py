e = 2.718281828459045

def sigmoid(z):
    return 1/(1+e**(0-z))

def relu(z):
    return max(0.01*z,z)
