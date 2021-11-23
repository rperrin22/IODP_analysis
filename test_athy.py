import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# load in the porosity data from the two logs
filename1417 = '/home/rperrin/Documents/IODP_porosity_analysis/341_U1417E/341-U1417E_apsm.dat'
D = pd.read_csv(filename1417,skiprows=[0,1,2,3,5],sep='\t')


# build the athy function
def rp_athy(mult,coeff,depth):
	por = mult*np.exp(coeff*depth)
	return por

xvec = np.arange(0,10,0.01)

mult = 0.7136
coeff = -0.3714

mult1 = 0.6136
mult2 = 0.7136
mult3 = 0.8136
mult4 = 0.9136
mult5 = 0.9855

yvec1 = rp_athy(mult1,coeff,xvec)
yvec2 = rp_athy(mult2,coeff,xvec)
yvec3 = rp_athy(mult3,coeff,xvec)
yvec4 = rp_athy(mult4,coeff,xvec)
yvec5 = rp_athy(mult5,coeff,xvec)

coeff1 = -0.3714
coeff2 = -0.2714
coeff3 = -0.4714

yvec6 = rp_athy(mult,coeff1,xvec)
yvec7 = rp_athy(mult,coeff2,xvec)
yvec8 = rp_athy(mult,coeff3,xvec)

plt.subplot(121)
plt.scatter(D.APLC/100,D.DEPTH_WMSF/1000,marker='^',s=1,c='black')
plt.plot(yvec1,xvec,label='mult = %.4f' % mult1)
plt.plot(yvec2,xvec,label='mult = %.4f' % mult2)
plt.plot(yvec3,xvec,label='mult = %.4f' % mult3)
plt.plot(yvec4,xvec,label='mult = %.4f' % mult4)
plt.plot(yvec5,xvec,label='mult = %.4f' % mult5)
plt.title('Testing multiplier')
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.xlabel('Porosity')
plt.ylabel('Depth (km)')

plt.subplot(122)
plt.scatter(D.APLC/100,D.DEPTH_WMSF/1000,marker='^',s=1,c='black')
plt.plot(yvec6,xvec,label='coeff = %.4f' % coeff1)
plt.plot(yvec7,xvec,label='coeff = %.4f' % coeff2)
plt.plot(yvec8,xvec,label='coeff = %.4f' % coeff3)
plt.title('Testing exponent coeff')
plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.xlabel('Porosity')
plt.ylabel('Depth (km)')


plt.show()
