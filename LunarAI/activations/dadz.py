e = 2.718281828459045

def sigmoid(z): #d sigmoid/dz
    a = 1/(1+e**(0-z))
    return a*(1-a)

def relu(z):   #d relu/dz
    r = z.get()
    for i in range(len(z)):
        for j in range(len(i)):
            if z[i][j]>0:
                r[i][j] = 1
            else:
                r[i][j] =  0.01
            continue
        continue
    return r