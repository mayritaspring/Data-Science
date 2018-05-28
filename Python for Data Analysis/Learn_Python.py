#查看資料型態
b = [1, 2, 3]
b?

#函數
def f(x, y, z):
    return (x + y) / z
a = 5
b = 6
c = 7.5
result = f(a, b, c)

#%timeit
from numpy import np
a = np.random.randn(100, 100)
%timeit np.dot(a, a)

#利用tab功能查看能用的功能
a = 'foo'
a.capitalize()
getattr(a, 'split')

#符號使用
a = 1222
b = 89
a//b
a/b

#使用套件pandas
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
data.shape