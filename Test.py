
import scipy.spatial




print(scipy.spatial.distance.cdist([[1]], [[2]]))





array = []

for i in range(100):
    array.append(i)
    print(i)


print(array[10:50])


for i in range(0,len(array),32):
    print(array[i:i+32])