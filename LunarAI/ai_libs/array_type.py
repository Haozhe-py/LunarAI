from threading import Thread
try:
    from matrix import Matrix
except ImportError:
    from LunarAI.ai_libs.matrix import Matrix


class Array(Matrix) :
    def get(self, value = 0):
        return Array(super().get(value = value))
    #dot
    def dot(self , other):
        '''calculate the dot product of self·other '''
        return super().__mul__(other)
    #add
    def __add_helper(self , a , i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] + a)
            continue
        new[i] = line_i
        return None
    
    def __add__(self, other):
        if isinstance(other, Matrix) or isinstance(other, Array):
            return super().__add__(other)
        else:
            new = self.get()
            i_s,j_s = self.size()
            for i in range(i_s):
                Thread( 
                    target = self.__add_helper(other,i,new, j_s)
                ).start()
                continue
            return new
            
    def __iadd__(self , other):
        self = self + other
        return self

    #sub
    def __sub_helper(self , a , i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] - a)
            continue
        new[i] = line_i
        return None
    
    def __sub__(self, other):
        if isinstance(other, Matrix) or isinstance(other, Array):
            return super().__sub__(other)
        else:
            new = self.get()
            i_s,j_s = self.size()
            for i in range(i_s):
                Thread( 
                    target = self.__sub_helper(other,i,new, j_s)
                ).start()
                continue
            return new
            
    def __isub__(self , other):
        self = self - other
        return self
    
    #mul
    def __mul_helper(self , a , i , new, j_s):#mul helper for int/float/...
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] * a)
            continue
        new[i] = line_i
        return None
    def __mul_helper_2(self , a , i, new, j_s): #mul helper for array/matrix type
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] * a[i][j])
            continue
        new[i] = line_i
        return None
    def __mul__(self , other):
        new = self.get()
        i_s, j_s = self.size()
        if isinstance(other,Matrix) or isinstance(other , Array) :
            #mul for Matrix/Array
            for i in range(len(self)):
                Thread(target=lambda:self.__mul_helper_2(other,i,new,j_s)).start()
                continue
        else:
            #mul for int/float/...
            for i in range(len(self)):
                Thread(target=lambda:self.__mul_helper(other,i,new,j_s)).start()
                continue
        return new
    def __imul__(self , other):
        self = self * other
        return self

    #truediv
    def __truediv_helper(self , a , i , new, j_s):#truediv helper for int/float/...
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] / a)
            continue
        new[i] = line_i
        return None
    def __truediv_helper_2(self , a , i, new, j_s): #truediv helper for array/matrix type
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] / a[i][j])
            continue
        new[i] = line_i
        return None
    def __truediv__(self , other):
        new = self.get()
        i_s, j_s = self.size()
        if isinstance(other,Matrix) or isinstance(other , Array) :
            #truediv for Matrix/Array
            for i in range(len(self)):
                Thread(target=lambda:self.__truediv_helper_2(other,i,new,j_s)).start()
                continue
        else:
            #truediv for int/float/...
            for i in range(len(self)):
                Thread(target=lambda:self.__truediv_helper(other,i,new,j_s)).start()
                continue
        return new
    def __itruediv__(self , other):
        self = self / other
        return self
    #pow
    def __pow_helper(self , a , i , new, j_s):#pow helper for int/float/...
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] ** a)
            continue
        new[i] = line_i
        return None
    def __pow_helper_2(self , a , i, new, j_s): #pow helper for array/matrix type
        line_i = []
        for j in range(j_s):
            line_i.append(self[i][j] ** a[i][j])
            continue
        new[i] = line_i
        return None
    def __pow__(self , other):
        new = self.get()
        i_s, j_s = self.size()
        if isinstance(other,Matrix) or isinstance(other , Array) :
            #pow for Matrix/Array
            for i in range(len(self)):
                Thread(target=lambda:self.__pow_helper_2(other,i,new,j_s)).start()
                continue
        else:
            #pow for int/float/...
            for i in range(len(self)):
                Thread(target=lambda:self.__pow_helper(other,i,new,j_s)).start()
                continue
        return new
    def __ipow__(self , other):
        self = self / other
        return self
    #radd/rsub/rmul/rtruediv/rpow
    def __radd__(self, other):
        return self + other  #other + self = self + other
    def __rsub__(self, other):
        return (self * (-1)) + other #other - self = -self + other = (-1)*self + other
    def __rmul__(self, other):
        return self * other  #other * self = self * other
    
    def __rtruediv_helper(self, a, i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(a / self[i][j])
            continue
        new[i] = line_i
        return None

    def __rtruediv__(self, other):
        new = self.get()
        i_s, j_s = self.size()
        for i in range(len(self)):
            Thread(
                target=lambda: self.__rtruediv_helper(other, i, new, j_s)
            ).start()
            continue
        return new

    def __rpow_helper(self, a, i, new, j_s):
        line_i = []
        for j in range(j_s):
            line_i.append(a ** self[i][j])
            continue
        new[i] = line_i
        return None

    def __rpow__(self, other):
        new = self.get()
        i_s, j_s = self.size()
        for i in range(len(self)):
            Thread(
                target=lambda: self.__rpow_helper(other, i, new, j_s)
            ).start()
            continue
        return new
