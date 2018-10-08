import glob
import os 
from matplotlib import cm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#chen 
path =r'C:\Users\delbe\Downloads\wut\wut\FALL_2018\Post_grad\accili\2018\lab\digitize\chen2007\fig8B\new csv' 

act_files = glob.glob(path + "/*left*.csv")
de_files = glob.glob(path + "/*right*.csv")

act_num = len(act_files)
de_num = len(de_files) 


files = act_files

d = {}
l = []
cols = []

nd_act = {} 
nd_de = {} 

ddact = [] 
aleg = []
for f in act_files: 
    prefix = os.path.basename(os.path.normpath(f)) #get the filename without directory/path 
    short_prefix = prefix[-11:-5] #filename
    aleg.append(str(short_prefix))
    df = pd.read_csv(f) 
    ddact.append(df) 

dlist = [] 
for da in ddact: 
    df = pd.DataFrame.from_dict(da) 
    dlist.append(df) 
#dfMT.to_excel('merged_act_files_4s.xlsx')    

dleg = [] 
for f in de_files:
    prefix = os.path.basename(os.path.normpath(f)) #get the filename without directory/path 
    short_prefix = prefix[-10:-5] #filename
    dleg.append(short_prefix)
    df = pd.read_csv(f) 
    dlist.append(df)
    
dfMerge = pd.concat(dlist, axis=1, sort=True)
dfMT = dfMerge.T 

#unbound params 
a0=0.0000032
sa=9.1
b0=415.8
sb=49
c0=4e-7 #a0'
sc=8.3 #sa'
d0=0.045 #b0'
sd=33.9 #sb' 
g1=0.0024
h1=0.00038
g2=0.0000042
h2=0.021
'''
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
'''
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
    #a1=(a2*g2*h1*b1)/(b2*h2*g1) 
    
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = [w for w in W]
        
    SS_4s = [0.9997124589964191, 1.4115735053476677e-06, 0.00019851847336416602, 8.761095671133587e-05]

    const = np.linalg.solve(V, SS_4s)

    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 8005, 5):
        cr4s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 4))
        ca4s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 4))
        or4s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 4))
        oa4s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 4))

        #totals
        C_t = cr4s + ca4s 
        O_t = or4s + oa4s 

        ss_l.update( {x : O_t} ) 

    nss.update( {vol : ss_l} )    

#deactivation 
for vol in [-10, 10, 40]:
    #unbound 
    #a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)
    a1=(a2*g2*h1*b1)/(b2*h2*g1) 
    
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    
    SS_4s = [0.1748911696847335, 0.11254557581393589, 3.389336600555093e-06, 0.71255986516473]
    const = np.linalg.solve(V, SS_4s)

    #analytical simulations
    ss_l = {}
    for x in range(0, 405, 5):
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
print(dss)
f1 = plt.figure()
f2 = plt.figure() 

xval1 = dfMerge.iloc[:, [0, 2, 4, 6, 8, 10]] #WT activation
yval1 = dfMerge.iloc[:, [1, 3, 5, 7, 9, 11]] #WT activation
xval2 = dfMerge.iloc[:, [12, 14, 16]] #WT deactivation
yval2 = dfMerge.iloc[:, [13, 15, 17]] #WT deactivation

act_sim = dss.iloc[:, [0, 1, 2, 3, 4, 5]]
de_sim = dss.iloc[:, [6, 7, 8]]
volts = dss.columns.values.tolist()

ax1 = f1.add_subplot(111) #activation 
l1 = ax1.plot(xval1, yval1)
ax1.set_ylabel('Normalized open fraction') 
l2 = ax1.plot(act_sim, linestyle='--') 
ax1.set_xlabel('Time units (ms)')
#
sim_act_leg = ax1.legend(l2, volts[0:7], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0.)
leg3 = ax1.add_artist(sim_act_leg)
ax1.legend(l1, aleg, bbox_to_anchor=(1.11, 1), loc=1, borderaxespad=0.)

ax3 = f2.add_subplot(111) #deactivation
l3 = ax3.plot(xval2, yval2)
l4 = ax3.plot(de_sim, linestyle='--')
ax3.set_xlabel('Time units (ms)')
#
de_act_leg = ax3.legend(l4, volts[6:9], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0.)
leg4 = ax3.add_artist(de_act_leg)
ax3.legend(l3, dleg, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel("Time")
plt.ylabel("Normalized current")
ax1.set_title("WT activation", y=1.08)
ax3.set_title("WT deactivation", y=1.08)

#ax1.legend(aleg)
#ax2.legend(dleg)

plt.show() 
