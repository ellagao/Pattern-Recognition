import numpy as np
import matplotlib
from matplotlib import pyplot as plt

k,nc,coff,N=0,0,0.2,6
w_vector,x_vector=[],[]
w_vector.append(np.matrix('1 1 1'))
x_vector.append(np.matrix('1 0 1'))
x_vector.append(np.matrix('1 1 1'))
x_vector.append(np.matrix('-2 -1 -1'))
x_vector.append(np.matrix('-2 -2 -1'))
x_vector.append(np.matrix('0 2 1'))
x_vector.append(np.matrix('-1 -3 -1'))
ax = plt.gca()  # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none')  # 将右边 边沿线颜色设置为空 其实就相当于抹掉这条边
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置中心的为（0，0）的坐标轴
ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))

plt.xlim(-1.0, 4.0) #x轴数值设置
plt.ylim(-1.0, 4.0)
plt.plot(1, 0,  color='r',marker='o')
plt.plot(1, 1,  color='r',marker='o')
plt.plot(0, 2,  color='r',marker='o')
plt.plot(2, 1,  color='b',marker='o')
plt.plot(2, 2,  color='b',marker='o')
plt.plot(1, 3,  color='b',marker='o')
while(nc<N):
    kn=k%N
    GX=float(w_vector[k]*x_vector[kn].T)
    if GX>0.000001:
        nc+=1
        w_vector.append(w_vector[k])
    else:
        w_vector.append(w_vector[k]+coff*x_vector[kn])
        nc=0
    k+=1

for i in range(6):
    print('X'+str(i),end=' ')
    print(x_vector[i])
print('w[0] = ',end='')
print(w_vector[0])
print('w* = ',end='')
print(w_vector[k])
print('coff = '+str(coff))
print('times = '+str(k))
xx=float(w_vector[k].T[0])
yy=float(w_vector[k].T[1])
zz=float(w_vector[k].T[2])
kx=-xx/yy
bx=-zz/yy
x = np.linspace(-4,4,100)
y=kx*x+bx
print('G(x): %.2f x + %.2f y + %.2f =0'%(xx,yy,zz))
plt.plot(x,y)
plt.title('seq: + + - - + -')
plt.show()
