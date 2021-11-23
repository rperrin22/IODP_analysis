from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# build the athy function
def rp_athy(mult,coeff,depth):
    por = mult*np.exp(coeff*depth)
    return por

# create function to make this plot
def plot_vert_profiles(param_file,xpos,ypos):
    """
        plot vertical temperature and pressure profiles for a given location
        in the survey area.  
    """
    D = pd.read_csv(param_file)

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    for x in range(len(D)):
        filename = '%s/%s.36500000.0_days_sca_node.csv' % (D.prefix[x],D.prefix[x])
        D1 = pd.read_csv(filename)
        DS1 = D1[(D1[' X coordinate (m)']==xpos) & (D1[' Y coordinate (m)']==ypos)]
        plt.plot(DS1[' Temperature (deg C)'],DS1[' Z coordinate (m)'],'*:',label='Athy mult = %s'%D.athy_multiplier[x])

    plt.grid()
    plt.legend()
    plt.ylabel('Elevation (m)')
    plt.xlabel('Temperature (deg C)')

    D2 = pd.read_csv(D.surf_filename[0],sep=' ',names=['x','y','z'])
    triang = tri.Triangulation(D2.y, D2.x) # backwards because of order=F, reverse if created in Python

    D2.z = D2.z+D.zbulk[0]

    plt.subplot(2,2,3)
    xvec = np.arange(D.min_x[0],D.max_x[0],D.dx[0])
    yvec = np.arange(D.min_y[0],D.max_y[0],D.dy[0])
    zvec = np.arange(0,D.upper_numlayers[0]+D.middle_numlayers[0]+D.lower_numlayers[0],1)
    XX,YY,ZZ = np.meshgrid(xvec,yvec,zvec)

    for x in range(len(D)):
        filename = '%s/%s.36500000.0_days_sca_node.csv' % (D.prefix[x],D.prefix[x])
        D1 = pd.read_csv(filename)
        D1['ZZtemp'] = ZZ.flatten(order='F')
        D_temp = D1[D1.ZZtemp==12]
        D_temp = D_temp[D1[' X coordinate (m)']==xpos].copy()
        plt.plot(D_temp[' Y coordinate (m)'],D_temp[' Temperature (deg C)'],label='mult = %s'%D.athy_multiplier[x])

    plt.grid()
    plt.legend()
    plt.xlabel('Y coordinate (m)')
    plt.ylabel('Temperature (deg C)')    

    plt.subplot(2,2,2)
    levels = np.arange(min(D2.z), max(D2.z), 10)
    cmap = cm.get_cmap(name='terrain', lut=None)
    tcf = plt.tricontourf(triang, D2.z, levels=levels, cmap=cmap)
    plt.tricontour(triang, D2.z, levels=levels,
                colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])

    plt.scatter(x=xpos,y=ypos,c='k')
    plt.plot(D_temp[' X coordinate (m)'],D_temp[' Y coordinate (m)'],c='black')
    plt.title("Crust Surface")
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    cbar = plt.colorbar(tcf)
    cbar.set_label('Elevation (m)')

    plt.subplot(2,2,4)
    # load in the porosity data from the two logs
    filename1417 = '/home/rperrin/Documents/IODP_porosity_analysis/341_U1417E/341-U1417E_apsm.dat'
    Dasdf = pd.read_csv(filename1417,skiprows=[0,1,2,3,5],sep='\t')


    plt.scatter(Dasdf.APLC/100,Dasdf.DEPTH_WMSF/1000,marker='^',s=1,c='black')

    xvec = np.arange(0,10,0.01)
    coeff = -0.3714

    for x in range(len(D)):
        mult=D.athy_multiplier[x]
        yvec=rp_athy(mult,coeff,xvec)
        plt.plot(yvec,xvec,label='mult = %.4f' % mult)

    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()
    plt.xlabel('Porosity')
    plt.ylabel('Depth (km)')
    plt.xlim([0,1])


    plt.show()

plot_vert_profiles('param_file_new.csv',6000,6000)

