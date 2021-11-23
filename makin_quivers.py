from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# load in the file yo
filename = 'test1/test1.36500000.0_days_vec_node.csv'
D = pd.read_csv(filename)


ax = plt.figure().add_subplot(projection='3d')
ax.quiver(D['  X Coordinate (m)'],
          D['  Y Coordinate (m)'],
          D['  Z Coordinate (m)'],
          D['  Liquid X Volume Flux (m3/[m2 s])'],
          D['  Liquid Y Volume Flux (m3/[m2 s])'],
          D['  Liquid Z Volume Flux (m3/[m2 s])'],
           length=10000000000, normalize=False)
plt.show()