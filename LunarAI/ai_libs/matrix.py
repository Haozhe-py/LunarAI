from threading import Thread

def check(__list) -> None:
    l = len(__list[0])
    for i in __list:
        if len(i) != l:
            raise ValueError
        else:
            continue
        continue
    return None

class Matrix(list):
    def __init__(self,value) -> None:
        super().__init__(value)
        check(self)
        return None
    def size(self):
        return len(self),len(self[0])
    def get(self , value = 0) :
        i_s , j_s = self.size()
        new = [
            [value] * j_s
        ] * i_s
        return new
    def __str__(self) -> str:
        s = ''
        for i in self:
            s += str(i)[1:-1]
            s += '\n'
            continue
        return s
    def __call__(self,i,j):
        return self[i][j]
    #add
    def __add_helper(self , a , i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] + a[i][j])
            continue
        new[i] = line_i
    def __add__(self, other) :
        if not isinstance(other,Matrix):
            raise TypeError
        if self.size() != other.size():
            raise ValueError
        new = self.get()
        i_s,j_s = self.size()
        for i in range(i_s):
            Thread( 
                target = self.__add_helper(other,i,new, j_s)
            ).start()
            continue
        return new
    def __iadd__(self, other) :
        self = self + other
        return self
    #sub
    def __sub_helper(self , a , i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] - a[i][j])
            continue
        new[i] = line_i
    def __sub__(self, other) :
        if not isinstance(other,Matrix):
            raise TypeError
        if self.size() != other.size():
            raise ValueError
        new = self.get()
        i_s,j_s = self.size()
        for i in range(i_s):
            Thread(
                target=lambda:self.__sub_helper(other,i,new,j_s)
            ).start()
            continue
        return new
    def __isub__(self, other) :
        self = self - other
        return self
    #mul
    def __mul_helper(self,other,new,i,k):
        new_ik = 0
        i_s , j_s = self.size()
        for j in range(j_s):
            new_ik += self[i][j]*other[j][k]
            continue
        new[i][k] = new_ik
        return None
    def __mul__(self, other):
        s_i , s_j = self.size()
        o_i , o_j = other.size()
        if s_j != o_i :
            raise ValueError
        new = Matrix([[0for k in range(o_j)] for i in range(s_i)])
        for i in range(s_i):
            for k in range(o_j):
                Thread(target=lambda:self.__mul_helper(other,new,i,k)).start()
                continue
            continue
        return new
    def __imul__(self, other):
        self = self * other
        return self
    #Matrix.T
    def __T_helper(self,new,i):
        j_size = len(self[0])
        for j in range(j_size):
            new[j][i] = self[i][j]
            continue
        return
    @property
    def T(self):
        i_size ,j_size = self.size()
        new = Matrix( [ [0 for j in range(i_size)] for i in range(j_size)])
        for i in range(i_size):
            Thread(target = lambda:self.__T_helper(new,i)).start()
            continue
        return new
'''
a=Matrix( [
    [3,31],
    [12,2],
    [1,12]
])
b=Matrix( [
    [22,12,27],
    [53,73,35]
])
c=Matrix( [
    [3,7],
    [2,23],
    [1,9999]
])

print(a.T)
#a+c

'''
