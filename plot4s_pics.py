import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.misc import imread
import matplotlib.cbook as cbook
from PIL import Image 
import pandas as pd 

#unbound params 
a0=0.0000032
sa=9.1
b0=415.8
sb=49
c0=0.00011 #a0'
sc=9.9 #sa'
d0=0.045 #b0'
sd=33.9 #sb' 
g1=0.0024
h1=0.00038
g2=0.0000042
h2=0.021

nd4s = {}

nss = {} 
eigVec = {} 
eigVal = {} 

for vol in range(-105, -165, -10):
    
    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)
    
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    print(Q.shape)
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    print(EigD)
    print(W.shape)
    
    #activation - held at -40, deactivation held at -140 
    #unbound 
    v=-40
    a1=a0*np.exp(-v/sa) 
    b1=b0*np.exp(v/sb) 
    a2=c0*np.exp(-v/sc) 
    b2=d0*np.exp(v/sd)
    
    #LHS terms
    alpha = a1 + g2 
    beta = b1 + g1
    gamma = b2 + h1 
    delta = a2 + h2 
    
    #4s 23/8/18
    #x
    x1 = beta - ((a1*b1)/alpha)
    x2 = (a1*h2)/alpha 
    
    x3 = delta - ((g2*h2)/alpha) 
    x4 = (b1*g2)/alpha 
    
    #y
    y1 = gamma - h1*(g1/x1) 
    y2 = a2 + x2*(g1/x1) 
    y3 = y1/y2 
    
    y4 = ((x3*y3 - b2)/x4) 
    
    y5 = b1*y4 + h2*y3 
    y6 = y5/alpha 
    
    #eqns 
    o4_4s = 1/(1 + y6 + y4 + y3) 
    c1_4s = y6*o4_4s
    c2_4s = y4*o4_4s
    o3_4s = y3*o4_4s
        
    d4s = {} 
    d4s = { 'c1' : c1_4s, 'c2' : c2_4s, 'o3' : o3_4s, 'o4' : o4_4s}
    nd4s.update({ vol : d4s })
    
    SS_4s = [c1_4s, c2_4s, o3_4s, o4_4s]
    const = np.linalg.solve(V, SS_4s)
    
    print('E')
    print(EigD)
    
    print('det')
    det = np.linalg.det(Q)
    print(det)
    
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 8001, 1):
        cr4s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 4))
        ca4s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 4))
        or4s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 4))
        oa4s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 4))

        #totals
        C_t = cr4s + ca4s 
        O_t = or4s + oa4s 

        #current
        #if1=1000000000*(gf1*O_ut + gf2*O_bt)*(vol-rev)
    
        #if_l.append(if1)
        ss_l.update( {x : O_t} ) 
        
    nss.update( {vol : ss_l} )    
    
#deactivation 
for vol in [-10, 10, 40]:
    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)
    
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    print(Q.shape)
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    print(EigD)
    print(W.shape)
    
    #activation - held at -40, deactivation held at -140 
    v = -140 
    #unbound 
    a1=a0*np.exp(-v/sa) 
    b1=b0*np.exp(v/sb) 
    a2=c0*np.exp(-v/sc) 
    b2=d0*np.exp(v/sd)
    
    #LHS terms
    alpha = a1 + g2 
    beta = b1 + g1
    gamma = b2 + h1 
    delta = a2 + h2 
    
    #4s 23/8/18
    #x
    x1 = beta - ((a1*b1)/alpha)
    x2 = (a1*h2)/alpha 
    
    x3 = delta - ((g2*h2)/alpha) 
    x4 = (b1*g2)/alpha 
    
    #y
    y1 = gamma - h1*(g1/x1) 
    y2 = a2 + x2*(g1/x1) 
    y3 = y1/y2 
    
    y4 = ((x3*y3 - b2)/x4) 
    
    y5 = b1*y4 + h2*y3 
    y6 = y5/alpha 
    
    #eqns 
    o4_4s = 1/(1 + y6 + y4 + y3) 
    c1_4s = y6*o4_4s
    c2_4s = y4*o4_4s
    o3_4s = y3*o4_4s
        
    d4s = {} 
    d4s = { 'c1' : c1_4s, 'c2' : c2_4s, 'o3' : o3_4s, 'o4' : o4_4s}
    nd4s.update({ vol : d4s })
    
    SS_4s = [c1_4s, c2_4s, o3_4s, o4_4s]
    const = np.linalg.solve(V, SS_4s)
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 401, 1):
        cr4s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 4))
        ca4s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 4))
        or4s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 4))
        oa4s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 4))

        #totals
        C_t = cr4s + ca4s 
        O_t = or4s + oa4s 

        ss_l.update( {x : O_t} ) 
    
    nss.update( {vol : ss_l} )  

last1 = nss[-155][8000]
for k in nss.keys(): 
    nss[k] = {key:(val/last1) for key, val in nss[k].items()}
last2 = nss[40][0]
for k in [-10, 10, 40]:
    nss[k] = {key:(val/last2) for key, val in nss[k].items()}
    
dss = pd.DataFrame.from_dict(nss)

#create figures 
f1 = plt.figure()
f2 = plt.figure() 

#model data
act_sim = dss.iloc[:, [0, 1, 2, 3, 4, 5]]
de_sim = dss.iloc[:, [6, 7, 8]]

actx = list(dss.index)
acty = act_sim
# 
#model legend from dataframe headings 
volts = dss.columns.values.tolist()

#initialize images to plot 
image_list = []
impath = r'C:/Users/delbe/Downloads/wut/wut/FALL_2018/Post_grad/accili/2018/lab/digitize/chen2007/original figs'
actim = glob.glob(impath + '/*left*_2*.png')
deim = glob.glob(impath + '/*right*post3*.png')
for imaj in actim:
    img = imread(imaj)
    image_list.append(img)
for imdj in deim:
    img = imread(imdj)
    image_list.append(img) 

#activation 
ax1 = f1.add_subplot(111)
#axes labels 
ax1.set_xlabel('Time units (ms)', fontsize=14)
ax1.set_ylabel('Normalized open fraction', fontsize=14)
#load images 
act_img = image_list[0]
#plots 
l1 = ax1.plot(actx, acty, zorder=2, linewidth=3)
l2 = ax1.imshow(act_img, zorder=1, aspect='auto', extent=[0, 8000, 0, 1], alpha=0.6)
#legends 
sim_act_leg = ax1.legend(l1, volts[0:7], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0., fontsize='large')
leg1 = ax1.add_artist(sim_act_leg)

#deactivation 
ax2 = f2.add_subplot(111) #deactivation
#axes labels 
ax2.set_xlabel('Time units (ms)', fontsize=14)
ax2.set_ylabel('Normalized open fraction', fontsize=14)
#image
de_img = image_list[1]
#plots 
l3 = ax2.plot(de_sim, zorder=2, linewidth=3)
l4 = ax2.imshow(de_img, zorder=1, aspect='auto', extent=[0, 400, 0, 1], alpha=1)
#legends 
de_act_leg = ax2.legend(l3, volts[6:9], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0., fontsize='large')
leg2 = ax2.add_artist(de_act_leg)

#figure titles 
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)
ax1.set_title("WT activation", y=1.02)
ax2.set_title("WT deactivation", y=1.02)
#
plt.show()