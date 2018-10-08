import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
import io 
import glob

path =r'C:/Users/delbe/Downloads/wut/wut/FALL_2018/Post_grad/accili/2018/lab/digitize/kim2012/test/WT' 

act_files = glob.glob(path + "/*act*.csv")
de_files = glob.glob(path + "/*de*.csv")

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
    short_prefix = prefix[16:-8] #filename
    aleg.append(str(short_prefix))
    df = pd.read_csv(f) 
    ddact.append(df) 

dlist = [] 
for da in ddact: 
    df = pd.DataFrame.from_dict(da) 
    dlist.append(df) 
#dfMT.to_excel('merged_act_files_6s.xlsx')    


dleg = [] 
for f in de_files:
    prefix = os.path.basename(os.path.normpath(f)) #get the filename without directory/path 
    short_prefix = prefix[15:-8] #filename
    dleg.append(short_prefix)
    df = pd.read_csv(f) 
    dlist.append(df)
    
dfMerge = pd.concat(dlist, axis=1, sort=True)

dfMT = dfMerge.T 

def frange(start, stop, step):
    s = start
    if step < 0:
        while s > stop:
            yield s
            s += step
            s = round(s, 15)
    if step > 0:
        while s < stop:
            yield s
            s += step
            s = round(s, 15)

#voltage-independent (1/s) 
g = 0.0005
h = 0.0002
g1 = 0.00025 
h1 = 0.0125 
g2 = 0.0002
h2 = 0.00005

nd6s = {}
nss = {} 
ndUBT = {} 
ndexT = {} 

for vol in [-120, -110, -90, -80, -60, -50]:
    a = (4e-7)*np.exp(-0.065*vol) #lam
    b = (1e-3)*np.exp(0.0138*vol) #del
    a1 = (2.5e-6)*np.exp(-0.045*vol) #lam1
    b1 = (2e-3)*np.exp(0.0579*vol) #del1 )  
    l = (4e-7)*np.exp(-0.065*vol)
    d = (1e-3)*np.exp(0.0138*vol)
    l1 = (2.5e-6)*np.exp(-0.045*vol)
    d1 = (2e-3)*np.exp(0.0579*vol)
    
    """
    #transition rate matrix 
    Q = np.array([
                 [-(a + g1), b, 0, h1, 0, 0],
                 [a, -(b + l + g), d, 0, h, 0],
                 [0, l, -(d + g2), 0, 0, h2],
                 [g1, 0, 0, -(a1 + h1), b1, 0],
                 [0, g, 0, a1, -(b1 + l1 + h), d1],
                 [0, 0, g2, 0, l1, -(d1 + h2)]
                 ]
                 )
    """
    
    Q = np.array([
    [ -(a+g1),    b,      0,       0,     0,        h1],
    [a,    -(b+l+g),  d,       0,     h,        0], 
    [0,       l,    -(d+g2),  h2,   0,        0],
    [0,       0,     g2,  -(d1+h2), l1,       0],
    [0,       g,      0,      d1,  -(b1+l1+h),  a1],
    [g1,      0,      0,       0,    b1,   -(a1+h1)]
    ])
             
    W, V = np.linalg.eig(Q.T) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    
    v = 40 #hold at 0 for activation 
    a = (4e-7)*np.exp(-0.065*v) #lam
    b = (1e-3)*np.exp(0.0138*v) #del
    a1 = (2.5e-6)*np.exp(-0.045*v) #lam1
    b1 = (2e-3)*np.exp(0.0579*v) #del1   
    l = (4e-7)*np.exp(-0.065*v)
    d = (1e-3)*np.exp(0.0138*v)
    l1 = (2.5e-6)*np.exp(-0.045*v)
    d1 = (2e-3)*np.exp(0.0579*v)
   
    #LHS
    alpha = a + g1
    beta = b + g + l 
    gamma = d + g2 
    #delta = b1 + h2 
    delta = d1 + h2 
    epsil = b1 + l1 + h
    phi = a1 + h1 #FF

    #6s 30/8/18
    x1 = beta - b*(a/alpha) 
    x2 = h*(a/alpha) 
    
    x3 = delta - h*(g1/alpha)
    x4 = b*(g1/alpha)
    
    x5 = phi - h2*(g2/gamma) 
    x6 = a*(g2/gamma) 
    
    x7 = x1 - a*(b/gamma)
    x8 = b*(h2/gamma) 
    
    y1 = x7 - x4*(x2/x3)
    y2 = h + b1*(x2/x3)
    
    y3 = epsil - b1*(a1/x3) 
    y4 = g + x4*(a1/x3) 
    
    y5 = y1 - x6*(x8/x5) 
    y6 = y2 + a1*(x8/x5) 
    y7 = y5/y6 
    
    y8 = (a1*y7 + x6)/x5 
    
    y9 = (b1*y7 + x4)/x3 
    
    y10 = (a + h2*y8)/gamma 
    
    y11 = (b + h*y9)/alpha 
    
    #equations 
    c1_6s = 1/(y11 + 1 + y10 + y9 + y7 + y8) 
    cr_6s = y11*c1_6s
    o1_6s = y7*c1_6s
    c2_6s = y10*c1_6s
    or_6s = y9*c1_6s
    o2_6s = y8*c1_6s
    
    d6s = {} 
    d6s = { 'cr' : cr_6s, 'c1' : c1_6s, 'c2' : c2_6s, 'or' : or_6s, 'o1' : o1_6s, 'o2' : o2_6s}
    nd6s.update({ vol : d6s })

    SS_6s = [cr_6s, c1_6s, c2_6s, o2_6s, o1_6s, or_6s]
    const = np.linalg.solve(V, SS_6s)
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 8001, 1):
        cr6s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 6))
        c16s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 6))
        c26s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 6))
        o26s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 6))
        o16s = sum(const[i] * np.exp(EigD[i]*x) * V[4,i] for i in range(0, 6))
        or6s = sum(const[i] * np.exp(EigD[i]*x) * V[5,i] for i in range(0, 6))

        #totals
        C_t = cr6s + c16s + c26s 
        O_t = or6s + o16s + o26s
        
        #current
        #if1=1000000000*(gf1*O_ut + gf2*O_bt)*(vol-rev)
    
        #if_l.append(if1)
        ss_l.update( {x : O_t} ) 
    
    top = max((ss_l.values()))
    base = min(ss_l.values())
    
    nss.update( {vol : ss_l} )    

#deactivation 
for vol in [20, 0]:
    a = (4e-7)*np.exp(-0.065*vol) #lam
    b = (1e-3)*np.exp(0.0138*vol) #del
    a1 = (2.5e-6)*np.exp(-0.045*vol) #lam1
    b1 = (2e-3)*np.exp(0.0579*vol) #del1 )  
    l = (4e-7)*np.exp(-0.065*vol)
    d = (1e-3)*np.exp(0.0138*vol)
    l1 = (2.5e-6)*np.exp(-0.045*vol)
    d1 = (2e-3)*np.exp(0.0579*vol)
    
    #transition rate matrix 
    Q = np.array([
    [ -(a+g1),    b,      0,       0,     0,        h1],
     [a,    -(b+l+g),  d,       0,     h,        0], 
     [0,       l,    -(d+g2),  h2,   0,        0],
     [0,       0,     g2,  -(d1+h2), l1,       0],
     [0,       g,      0,      d1,  -(b1+l1+h),  a1],
     [g1,      0,      0,       0,    b1,   -(a1+h1)]
     ])
     
    W, V = np.linalg.eig(Q.T) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    
    v = -100 #hold at -100 for deactivation
    a = (4e-7)*np.exp(-0.065*v) #lam
    b = (1e-3)*np.exp(0.0138*v) #del
    a1 = (2.5e-6)*np.exp(-0.045*v) #lam1
    b1 = (2e-3)*np.exp(0.0579*v) #del1 )  
    l = (4e-7)*np.exp(-0.065*v)
    d = (1e-3)*np.exp(0.0138*v)
    l1 = (2.5e-6)*np.exp(-0.045*v)
    d1 = (2e-3)*np.exp(0.0579*v)
    
    #LHS
    FF = a + g1 
    alpha = b + g + l
    beta = d + g2 
    gamma = d1 + h2 
    delta = l1 + b1 + h
    epsil = a1 + h1 
    
    #6s_n2 
    x1 = b/FF
    x2 = h1/FF 
    
    y1 = alpha - a*x1 
    y2 = a*x2 
    y3 = epsil - g1*x2 
    y4 = g1*x1 
    
    y5 = g/delta 
    y6 = d1/delta
    y7 = a1/delta 
    
    z1 = y3 - b1*y7 
    z2 = y4 + b1*y5 
    
    z3 = gamma - l1*y6 
    z4 = l1*y5 
    z5 = l1*y7 
    
    z6 = y1 - h*y5 
    z7 = y2 + h*y7 
    z8 = h*y6 
    
    n1 = l/beta 
    n2 = h2/beta 
    
    n3 = z6 - d*n1 
    n4 = z8 + delta*n2 
    n5 = z3 - g2*n2 
    n6 = z4 + g2*n1 
    
    m1 = z1-z2*(z7/n3) 
    m2 = y6 + z2*(n4/n3) 
    m3 = m2/m1 
    
    m4 = n5 - z5*m3 
    m5 = m4/n6 
    
    m6 = y5*m5 + y6 + y7*m3 
    m7 = n1*m5 + h2 
    m8 = (1/FF)*(b*m5 + h1*m3) 
    
    #equations
    o2_6s = 1/(m8 + m5 + m7 + m3 + m6 + 1)
    cr_6s = m8*o2_6s
    c1_6s = m5*o2_6s
    c2_6s = m7*o2_6s
    or_6s = m3*o2_6s
    o1_6s = m6*o2_6s
    
    d6s = {} 
    d6s = { 'cr' : cr_6s, 'c1' : c1_6s, 'c2' : c2_6s, 'or' : or_6s, 'o1' : o1_6s, 'o2' : o2_6s}
    nd6s.update({ vol : d6s })

    SS_6s = [cr_6s, c1_6s, c2_6s, o2_6s, o1_6s, or_6s]
    const = np.linalg.solve(V, SS_6s)
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 401, 1):
        cr6s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 6))
        c16s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 6))
        c26s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 6))
        o26s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 6))
        o16s = sum(const[i] * np.exp(EigD[i]*x) * V[4,i] for i in range(0, 6))
        or6s = sum(const[i] * np.exp(EigD[i]*x) * V[5,i] for i in range(0, 6))

        #totals
        C_t = cr6s + c16s + c26s 
        O_t = or6s + o16s + o26s 

        ss_l.update( {x : O_t} ) 

    nss.update( {vol : ss_l} )   


last1 = nss[-120][8000]
for k in [-50, -60, -80, -90, -110, -120]: 
    for key, val in nss[k].items():
        val = (val - nss[k][0])/last1 
        
   #nss[k] = {key: (val - nss[k][0]) for key, val in nss[k].items()}
    #nss[k] = {key:(val/last1) for key, val in nss[k].items()}
    
"""
last2 = nss[20][0]
for k in [0, 20]:
    nss[k] = {key:(val/last2) for key, val in nss[k].items()}
"""
    
dss = pd.DataFrame.from_dict(nss)
dss.to_excel('OT_6s_kim.xlsx')

f1 = plt.figure()
f2 = plt.figure() 

xval1 = dfMerge.iloc[:, [0, 2, 4, 6, 8, 10]] #WT activation
yval1 = dfMerge.iloc[:, [1, 3, 5, 7, 9, 11]] #WT activation
xval2 = dfMerge.iloc[:, [12, 14]] #WT deactivation
yval2 = dfMerge.iloc[:, [13, 15]] #WT deactivation

act_sim = dss.iloc[:, [0, 1, 2, 3, 4, 5]]
de_sim = dss.iloc[:, [6, 7]]
volts = dss.columns.values.tolist()

ax1 = f1.add_subplot(111) #activation 
l1 = ax1.plot(xval1, yval1)
ax1.set_ylabel('Normalized open fraction') 
#l2 = ax1.plot(act_sim, linestyle='--') 
ax1.set_xlabel('Time units (ms)')

ax2 = f2.add_subplot(111)
l2 = ax2.plot(act_sim)

sim_act_leg = ax1.legend(l2, volts[0:6], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0.)
leg3 = ax1.add_artist(sim_act_leg)
ax1.legend(l1, aleg, bbox_to_anchor=(1.11, 1), loc=1, borderaxespad=0.)


ax3 = f2.add_subplot(111) #deactivation
l3 = ax3.plot(xval2, yval2)
l4 = ax3.plot(de_sim, linestyle='--')
ax3.set_xlabel('Time units (ms)')

de_act_leg = ax3.legend(l4, volts[6:8], bbox_to_anchor=(1.075, 0.80), loc=1, borderaxespad=0.)
leg4 = ax3.add_artist(de_act_leg)
ax3.legend(l3, dleg, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

'''
plt.xlabel("Time")
plt.ylabel("Normalized current")
ax1.set_title("WT activation", y=1.08)
ax3.set_title("WT deactivation", y=1.08)
''' 

#ax1.legend(aleg)
#ax2.legend(dleg)

plt.show()
