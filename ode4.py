import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def f(y, t, params):
    cr, c1, c2, opr, o1, o2 = y #set the values of y, a vector comprised of elements that correspond to each of the model's 6 states
    alpha, beta, alphap, betap, h, g, hp, gp, hpp, gpp = params #unpack parameters
    derivs = [-(gp + alpha)*cr + beta*c1 + hp*opr, -(beta + alpha + g)*c1 + alpha*cr + beta*c2 + h*o1, -(beta + gpp)*o2 + alpha*c1 + hpp*o2, -(hp + alphap)*opr + gp*cr + betap*o1, -(betap + h + alphap)*o1 + alphap*opr + g*c1 + betap*o2, -(hpp + betap)*o2 + alphap*o1 + gpp*c2]
    return derivs

# set parameters
v = -70; 
x1=1 * 10**-6; 
x2=0.063; 
x3=2.5 * 10**-6; 
x4=0.045; 
x5=0.002; 
x6=0.00115; 
x7=0.00003; 
x8=0.0336; 
alpha = x1*math.exp(-x2*v);  
alphap = x3*math.exp(-x4*v);   
beta = x5*math.exp(x6*v);  
betap = x7*math.exp(x8*v); 
g = 5 * 10**-4;  
gp = 0.57 * 10**-6;  
gpp =20 * 10**-4;                       
h = 1.5 * 10**-4;  
hp = 0.57 * 10**-4; 
hpp = 0.65 * 10**-4;

#initial values
cr0 = 1.0 
c10 = 0.0 
c20 = 0.0 
opr0 = 0.0
o10 = 0.0
o20 = 0.0

#bundle parameters for ODE solver
params = [alpha, alphap, beta, betap, g, gp, gpp, h, hp, hpp]

#bundle initial conditions for ODE solver
y0 = [cr0, c10, c20, opr0, o10, o20]

#Make time array for solution
tStop = 1000.
tInc = 1.0
t = np.arange(0., tStop, tInc)

#Call the ODE solver
psoln = odeint(f, y0, t, args=(params,))

#set figure
fig = plt.figure()
ax = fig.gca(projection='3d')

#set xyz variables
x = t
y = psoln[:,0]
z = psoln[:,1]
col = psoln[:,2]

#create meshgrids
X = x
Z, X = np.meshgrid(z, x)
Y, X = np.meshgrid (y, x)
COL, X = np.meshgrid (col, x)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='none', rstride=1, cstride=1, facecolors=cm.jet(COL/float(COL.max())))

#label axes
ax.legend()
mpl.rcParams['legend.fontsize'] = 10
ax.set_xlabel('time')
ax.set_ylabel('Cr')
ax.set_zlabel('C1')
ax.set_title('evolution of closed states over time');

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
