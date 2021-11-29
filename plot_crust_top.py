from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

D = pd.read_csv('param_file_new.csv')

filename = '%s/%s.36500000.0_days_sca_node.csv' % (D.prefix[0],D.prefix[0])

D1 = pd.read_csv(filename)
xvec = np.arange(D.min_x[0],D.max_x[0],D.dx[0])
yvec = np.arange(D.min_y[0],D.max_y[0],D.dy[0])
zvec = np.arange(0,D.upper_numlayers[0]+D.middle_numlayers[0]+D.lower_numlayers[0],1)
XX,YY,ZZ = np.meshgrid(xvec,yvec,zvec)

D1['ZZtemp'] = ZZ.flatten(order='F')
D_temp = D1[D1.ZZtemp==12]

plt.scatter(D_temp[' X coordinate (m)'], D_temp[' Y coordinate (m)'],c = D_temp[' Temperature (deg C)'])
plt.colorbar()
plt.show()
