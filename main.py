import numpy as np

arr = []
arr.append( 25*25*40/np.exp(0.5*0.5))
arr.append( 25*25*40/np.exp(0.5*1))
arr.append( 25*25*40/np.exp(0.5*1.5))
arr.append( 25*25*40/np.exp(0.5*2))
print(sum(arr))


def ro(t,x,b):
    return t/np.exp(x*b)

def d_ro(t,x,b):
    return -x*t/np.exp(x*b)

arr = []
arr.append(25*d_ro(40,0.5,0.5)/(0.2*ro(40,0.5,0.5)))
arr.append(25*d_ro(40,1,0.5)/(0.2*ro(40,1,0.5)))
arr.append(25*d_ro(40,1.5,0.5)/(0.2*ro(40,1.5,0.5)))
arr.append(25*d_ro(40,2,0.5)/(0.2*ro(40,2,0.5)))

print(sum(arr))

arr = []
arr.append(25*np.power(d_ro(40,0.5,0.5),2)*(.5/np.power(ro(40,0.5,0.5),2)+25/ro(40,0.5,0.5)))
arr.append(25*np.power(d_ro(40,1,0.5),2)*(.5/np.power(ro(40,1,0.5),2)+25/ro(40,1,0.5)))
arr.append(25*np.power(d_ro(40,1.5,0.5),2)*(.5/np.power(ro(40,1.5,0.5),2)+25/ro(40,1.5,0.5)))
arr.append(25*np.power(d_ro(40,2,0.5),2)*(.5/np.power(ro(40,2,0.5),2)+25/ro(40,2,0.5)))
print(sum(arr))
