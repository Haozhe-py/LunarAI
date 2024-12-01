e = 2.718281828459045

def sigmoid(z): #d sigmoid/dz
    a = 1/(1+e**(0-z))
    return a*(1-a)

def relu(z):   #d relu/dz
    if z>0:
        return 1
    return 0.01