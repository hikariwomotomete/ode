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
b0=480
sb=49
c0=0.00011
sc=9.9
d0=0.045
sd=33.9
g1=0.0024
h1=0.00038
g2=0.0000042
h2=0.021

#bound params 
a02=1.3e-7
sa2=9.6
b02=1042.7
sb2=13.6
a03=1.7e-6
sa3=6.9
b03=0.055
sb3=29.7
gb=0.0036
hb=1.5e-4
gb2=1.6e-6
hb2=0.016

#binding
k1 = 0.3
k2 = 0.7

ndn8 = {}
nss = {} 

for vol in range(-105, -165, -10):
    
    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)

    #bound
    ab=a02*np.exp(-vol/sa2) 
    bb=b02*np.exp(vol/sb2)
    ab2=a03*np.exp(-vol/sa3)
    bb2=b03*np.exp(vol/sb3)    
       
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2 + k1), b1, h2, 0, k2, 0, 0, 0],
                 [a1, -(b1 + g1 + k1), 0, h1, 0, k2, 0, 0],
                 [g2, 0, -(a2+h2+k1), b2, 0, 0, k2, 0],
                 [0, g1, a2, -(b2 + h1 + k1), 0, 0, 0, k2], 
                 [k1, 0, 0, 0, -(ab + gb2 + k2), bb, hb2, 0],
                 [0, k1, 0, 0, ab, -(bb + gb + k2), 0, hb],
                 [0, 0, k1, 0, gb2, 0, -(ab2 + hb2 + k2), bb2],
                 [0, 0, 0, k1, 0, gb, ab2, -(bb2 + hb + k2)]
                 ]
                 )
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    print(EigD)
    print(W.shape)
    
    v = 0
    #unbound 
    a1=a0*np.exp(-v/sa) 
    b1=b0*np.exp(v/sb) 
    a2=c0*np.exp(-v/sc) 
    b2=d0*np.exp(v/sd)

    #bound
    ab=a02*np.exp(-v/sa2) 
    bb=b02*np.exp(v/sb2)
    ab2=a03*np.exp(-v/sa3)
    bb2=b03*np.exp(v/sb3) 

    #LHS terms
    alpha = a1 + g2 + k1 
    beta = b1 + g1 + k1 
    gamma = b2 + h1 + k1
    delta = a2 + h2 + k1
    epsil = ab + gb2 + k2 
    FF = bb + gb + k2 
    G = bb2 + hb + k2
    H = ab2 + hb2 + k2 
   
    #n8 21/8/18
    #x
    x1 = alpha - ((g2*h2)/delta) 
    x2 = (b2*h2)/delta 
    x3 = (h2*k2)/delta 
    
    """
    #c2 p1 
    x4 = a1 - ((k1*k2)/bb) 
    x5 = (k2*epsil)/bb 
    x6 = ((hb2*k2)/bb) 
    """
    """
    #c2 np1 
    x33 = beta - ((k1*k2 + g1*hb)/FF) - ((a2*b1*hb)/(FF*h2)) 
    x4 = a1 - ((a2*hb*alpha)/(FF*h2))
    x5 = ((ab*k2)/FF) + ((a2*hb*k2)/(FF*h2))
    x6 = 0 
    """
    #c2 np2 
    x33 = beta - ((k1*k2 - g1*hb)/FF)
    x4 = a1 - ((a2*g2*hb)/(FF*delta))
    x44 = h1 + ((hb*gamma)/FF) - ((a2*b2*hb)/(FF*delta))
    x5 = (ab*k2)/FF
    x6 = (a2*hb*k2)/(FF*delta) 
    
    #o4 by 3>3, 8>8>2
    x7 = gamma - ((a2*b2)/delta) - ((k1*k2)/G) + ((gb*h1)/G) 
    x8 = ((a2*g2)/delta) - ((a1*gb)/G)
    x9 = g1 + ((gb*beta)/G) 
    x10 = ((a2*k2)/delta) + ((ab2*k2)/G) 
    
    """
    #o4 by 8>7>3>3 
    x77 = a2 - ((k1*k2)/bb)
    x7 = gamma - b2*(x77/delta)
    x8 = g2*(x77/delta)
    x9 = g1 
    x10 = ((H*k2)/bb2) + k2*(x77/delta)
    """
    
    #c5 6>6>8>4 
    x11 = epsil - ((ab*bb)/FF)
    x12 = k1 - ((bb*hb*a2*g2)/delta) 
    x13 = ((k1*bb)/FF) - ((bb*g1*hb)/(FF*k2))
    x14 = ((bb*hb*gamma)/(FF*k2))
    x15 = hb2 - ((bb*hb*a2)/(FF*delta))
    
    """
    #c5 by 6>2 
    x11 = epsil 
    x12 = k1 - ((a1*bb)/k2)
    x13 = (bb*beta)/k2
    x14 = (h1*bb)/k2 
    x15 = hb2 
    """
    
    """
    #o7 by 8>4 
    x16 = H - ((k1*k2 - a2*bb2)/delta) 
    x17 = ((g2*k1*k2 - a2*bb2)/(delta*k2))
    x18 = (g1*bb2)/k2 
    x19 = ((gamma*bb2)/k2) + ((b2*k1*k2 - a2*bb2)/(k2*delta))
    """
    
    #o7 by 8>8>2, 3>3 
    th1 = H - ((ab2*bb2)/G) 
    th2 = (k1*bb2)/G 
    th3 = ((bb2*gb)/(G*k2)) 
    
    x16 = th1 - ((k1*k2)/delta) 
    x17 = ((g2*k1)/delta) - (a1*th3) 
    x18 = th3*beta 
    x19 = th2 - h1*th3 + ((b2*k1)/delta) 
    
    
    #y
    y1 = x1 - x8*(x2/x7) 
    y2 = b1 + x9*(x2/x7) 
    y3 = x3 + x10*(x2/x7) 
    
    """
    #c2 p1 
    #y4 = beta - x9*(h1/x7) 
    
    #c2 np1 
    y4 = x33 - x9*(h1/x7)
    
    y5 = x4 + x8*(h1/x7) 
    y6 = x10*(h1/x7) - x6
    """    
    
    #c2 np2 
    y4 = x33 - x9*(x44/x7)
    y5 = x4 + x8*(x44/x7) 
    y6 = x10*(x44/x7) - x6
    
    y7 = x12 + x8*(x14/x7) 
    y8 = x13 + x9*(x14/x7) 
    y9 = x15 + x10*(x14/x7) 
    
    y10 = x16 - x10*(x19/x7) 
    y11 = x17 + x8*(x19/x7) 
    y12 = x9*(x19/x7) - x18 
    
    #z 
    z1 = y1 - y5*(y2/y4)
    z2 = k2 + x5*(y2/y4)
    z3 = y3 + y6*(y2/y4)
    
    z4 = x11 - x5*(y8/y4)
    z5 = y7 + y5*(y8/y4)
    z6 = y9 + y6*(y8/y4)
    
    z7 = y10 - y6*(y12/y4)
    z8 = y11 + y5*(y12/y4)
    z9 = gb2 + x5*(y12/y4)
    
    #e
    e1 = z7 - z3*(z8/z1) 
    e2 = z9 + z2*(z8/z1) 
    e3 = e2/e1 
    
    e4 = ((z4 - z6*e3)/z5) 
    
    e5 = y5*e4 + x5 + y6*e3 
    e6 = e5/y4 
    
    e7 = x8*e4 + x9*e6 + x10*e3 
    e8 = e7/x7 
    
    e9 = g2*e4 + b2*e8 + k2*e3 
    e10 = e9/delta 
    
    e11 = ((gb/FF)*(ab + k1*e6)) + ab2*e3 + k1*e8 
    e12 = G - ((gb*hb)/FF) 
    e13 = e11/e12 
    
    e14 = ((hb/G)*(ab2*e3 + k1*e8)) + ab + k1*e6 
    e15 = FF - ((gb*hb)/G) 
    e16 = e14/e15 
    
    #equations n8 
    c5_n8 = 1/(1 + e4 + e6 + e10 + e8 + e16 + e3 + e13) 
    c1_n8  = e4*c5_n8 
    c2_n8 = e6*c5_n8 
    o3_n8 = e10*c5_n8 
    o4_n8 = e8*c5_n8 
    c6_n8 = e16*c5_n8 
    #c6_n8 = (1/2) - (1 + e13 + e3)*c5_n8 
    o7_n8 = e3*c5_n8 
    o8_n8 = e13*c5_n8 
    #o8_n8 = (1/2) - (1 + e16 + e3)*c5_n8 
    
    """
    #n8 (prb n3) 
    
    #theta 
    th1 = FF - ((ab*bb)/epsil) - ((k1*k2)/beta) - ((gb*hb)/G) 
    th2 = ((a1*k1)/beta) + ((ab*k1)/epsil) 
    th3 = ((h1*k1)/beta) + ((hb*k1)/G) 
    th4 = ((ab*hb2)/epsil) + ((ab2*hb)/G) 
    
    th5 = alpha - ((a1*b1)/beta) - ((g2*h2)/delta) - ((k1*k2)/epsil) 
    th6 = ((b1*h1)/beta) + ((b2*h2)/delta) 
    th7 = ((b1*k2)/beta) + ((bb*k2)/epsil)
    th8 = ((h2*k2)/delta) + ((hb2*k2)/epsil) 
    
    th9 = gamma - ((g1*h1)/beta) - ((a2*b2)/delta) - ((k1*k2)/G) 
    th10 = ((a1*g1)/beta) + ((a2*g2)/delta) 
    th11 = ((g1*k2)/beta) + ((gb*k2)/G) 
    th12 = ((a2*k2)/delta) + ((ab2*k2)/G) 
    
    th13 = H - ((gb2*hb2)/epsil) - ((ab2*bb2)/G) - ((k1*k2)/delta) 
    th14 = ((gb2*k1)/epsil) + ((g2*k1)/delta) 
    th15 = ((bb2*k1)/G) + ((b2*k1)/delta) 
    th16 = ((bb*gb2)/epsil) + ((bb2*gb)/G) 
    
    #A
    #x 
    x1 = th1 - th7*(th2/th5) 
    x2 = th3 + th6*(th2/th5) 
    x3 = th4 + th8*(th2/th5) 
    
    x4 = th9 - th6*(th10/th5) 
    x5 = th11 + th7*(th10/th5) 
    x6 = th12 + th8*(th10/th5) 
    
    x7 = th13 - th8*(th14/th5) 
    x8 = th16 + th7*(th14/th5) 
    x9 = th15 + th6*(th14/th5) 
    
    x10 = x4 - x9*(x6/x7) 
    x11 = x5 + x8*(x6/x7) 
    x12 = x11/x10 
    
    x13 = ((x8 + x12)/x7) 
    x14 = (1/th5)*(th6*x12 + th7 + th8*x13)
    
    x15 = (1/beta)*(a1*x14 + h1*x12 + k2) 
    
    x16 = (1/epsil)*(bb + hb2*x13 + k1*x12) 
    x17 = (1/delta)*(g2*x14 + b2*x12 + k2*x13) 
    x18 = (1/G)*(gb + ab2*x13 + k1*x12) 
   
    
    #equations (a) 
    c6_n8 = 1/(1 + x12 + x13 + x14 + x15 + x16 + x17 + x18)
    
    o4_n8 = x12*c6_n8
    o4_n8 = (1/2) - ((x14 + x15 + x17)*c6_n8)
    
    o7_n8 = x13*c6_n8
    o7_n8 = (1/2) - (c6_n8*(x16 + x13 + x18))
    
    c1_n8 = x14*c6_n8
    c2_n8 = x15*c6_n8
    c5_n8 = x16*c6_n8
    o3_n8 = x17*c6_n8
    o8_n8 = x18*c6_n8
    """
    
    dn8 = {} 
    dn8 = { 'c1' : c1_n8, 'c2' : c2_n8, 'o3' : o3_n8, 'o4' : o4_n8, 'c5' : c5_n8, 'c6' : c6_n8, 'o7' : o7_n8, 'o8' : o8_n8}
    ndn8.update({ vol : dn8 })
    
    SS_n8 = np.array([[c1_n8, c2_n8, o3_n8, o4_n8, c5_n8, c6_n8, o7_n8, o8_n8]])
    const = np.linalg.solve(V, SS_n8.T) 
    
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 8010, 10):
        cru = sum(const[i]*np.exp(EigD[i]*x)*V[0,i] for i in range(0, 8))
        cau = sum(const[i]*np.exp(EigD[i]*x)*V[1,i] for i in range(0, 8))
        oru = float(sum(const[i]*np.exp(EigD[i]*x)*V[2,i] for i in range(0, 8)).real)
        oau = float(sum(const[i]*np.exp(EigD[i]*x)*V[3,i] for i in range(0, 8)).real)
        crb = sum(const[i]*np.exp(EigD[i]*x)*V[4,i] for i in range(0, 8))
        cab = sum(const[i]*np.exp(EigD[i]*x)*V[5,i] for i in range(0, 8))
        orb = float(sum(const[i]*np.exp(EigD[i]*x)*V[6,i] for i in range(0, 8)).real)
        oab = float(sum(const[i]*np.exp(EigD[i]*x)*V[7,i] for i in range(0, 8)).real)
        
        #totals
        C_t = cru + cau + crb + cab
        O_t = float((oru + oau + orb + oab).real)
        U_t = cru + cau + oru + oau 
        B_t = crb + cab + orb + oau 
        O_ut = oau + oru 
        O_bt = oab + orb 

        ss_l.update( {x : O_t} ) 

    nss.update( {vol : ss_l} )     
       
#deactivation 
for vol in [-10, 10, 40]:
    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)

    #bound
    ab=a02*np.exp(-vol/sa2) 
    bb=b02*np.exp(vol/sb2)
    ab2=a03*np.exp(-vol/sa3)
    bb2=b03*np.exp(vol/sb3)  
       
    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2 + k1), b1, h2, 0, k2, 0, 0, 0],
                 [a1, -(b1 + g1 + k1), 0, h1, 0, k2, 0, 0],
                 [g2, 0, -(a2+h2+k1), b2, 0, 0, k2, 0],
                 [0, g1, a2, -(b2 + h1 + k1), 0, 0, 0, k2], 
                 [k1, 0, 0, 0, -(ab + gb2 + k2), bb, hb2, 0],
                 [0, k1, 0, 0, ab, -(bb + gb + k2), 0, hb],
                 [0, 0, k1, 0, gb2, 0, -(ab2 + hb2 + k2), bb2],
                 [0, 0, 0, k1, 0, gb, ab2, -(bb2 + hb + k2)]
                 ]
                 )

    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    
    #activation - held at -40, deactivation held at -140 
    v = -140 
    #unbound 
    a1=a0*np.exp(-v/sa) 
    b1=b0*np.exp(v/sb) 
    a2=c0*np.exp(-v/sc) 
    b2=d0*np.exp(v/sd)

    #bound
    ab=a02*np.exp(-v/sa2) 
    bb=b02*np.exp(v/sb2)
    ab2=a03*np.exp(-v/sa3)
    bb2=b03*np.exp(v/sb3) 

    #LHS terms
    alpha = a1 + g2 + k1 
    beta = b1 + g1 + k1 
    gamma = b2 + h1 + k1
    delta = a2 + h2 + k1
    epsil = ab + gb2 + k2 
    FF = bb + gb + k2 
    G = bb2 + hb + k2
    H = ab2 + hb2 + k2 
    
    """
    #n3
    #x
    x1 = b1 - ((k1*k2)/ab)
    x2 = (k2*FF)/ab
    x3 = (hb*k2)/ab
    x4 = beta - ((g1*h1)/gamma)
    x6 = (a2*h1)/gamma
    x7 = (h1*k2)/gamma
    x8 = delta - ((k1*k2)/H)
    x9 = ((b2*g1)/gamma) - ((gb2*k1*k2)/(ab*H))
    x10 = ((k1*k2)/H) + ((a2*b2)/gamma)
    x11 = ((gb2*k2*FF)/(ab*H))
    x12 = ((bb2*k2)/H) - ((gb2*hb*k2)/(ab*H)) + ((b2*k2)/gamma) 
    x13 = (ab*alpha)/k2 
    x14 = k1 - ((ab*b1)/k2)
    x15 = (ab*h2)/k2 
    x16 = G - (1/gamma)*(k1*k2 - ab2*b2)
    x17 = ((ab2*delta)/k2) - (1/(k2*gamma))*(a2*(k1*k2 - ab2*b2))
    x18 = ((ab2*g2)/k2)
    x19 = (g1/(k2*gamma))*(k1*k2 - ab2*b2) 

    #y
    y1 = alpha - x18*(x3/x16)
    y2 = x1 + x19*(x3/x16)
    y3 = h2 - x17*(x3/x16)
    y4 = x2 - gb*(x3/x16)
    y5 = x4 + x19*(x7/x16)
    y6 = a1 - x18*(x7/x16)
    y7 = x6 + x17*(x7/x16)
    y8 = k2 + gb*(x7/x16)
    y9 = x8 - x10 - x17*(x12/x16)
    y10 = g2 - x18*(x12/x16)
    y11 = x9 - x19*(x12/x16)
    y12 = x11 + gb*(x12/x16)
    y13 = FF - gb*(hb/x16) 
    y14 = x13 - x18*(hb/x16) 
    y15 = x14 - x19*(hb/x16) 
    y16 = x17*(hb/x16) - x15

    #d
    d1 = y1 - y6*(y2/y5)
    d2 = y3 + y7*(y2/y5)
    d3 = y4 + y5*(y2/y5)
    d4 = y9 - y7*(y11/y5)
    d5 = y10 + y6*(y11/y5)
    d6 = y12 + y8*(y11/y5)
    d7 = y13 - y8*(y15/y5)
    d8 = y14 + y6*(y15/y5)
    d9 = y16 + y7*(y15/y5)

    #e
    e1 = d4 - d2*(d5/d1)
    e2 = d6 + d3*(d5/d1)
    e3 = (1/d8)*(d7 - d9*(e2/e1))
    e4 = (1/y5)*(e3*y6 + y7*(e2/e1) + y8)
    e5 = (1/x16)*((x17*(e2/e1)) - x18*e3 + x19*e4 + gb)
    e6 = epsil - ((gb2*hb2)/H)
    e7 = bb + k1*e3 + ((e2/e1)*((k1*hb2)/H)) + e5*((bb2*hb2)/H)

    #f
    f1 = e5 + (e7/e6) + 1 
    f2 = e3 + e4 + (e2/e1) 
    f3 = (1/gamma)*(g1*e4 + a2*(e2/e1) + k2*e5)
    f4 = (1/H)*(gb2*(e7/e6) + bb2*(e5) + k1*(e2/e1))

    #equations
    c6_n8 = 1/(1 + e3 + e4 + (e2/e1) + f3 + (e7/e6) + f4 + e5)
    c1_n8 = e3*c6_n8
    c2_n8 = e4*c6_n8
    o3_n8 = (e2/e1)*c6_n8
    #o4_n8 = (1/2) - f2*c6_n8
    o4_n8 = f3*c6_n8
    c5_n8 = (e7/e6)*c6_n8 
    #o7_n8 = (1/2) - f1*c6_n8
    o7_n8 = f4*c6_n8
    o8_n8 = e5*c6_n8  
    """
    
     #n8 21/8/18
    #x
    x1 = alpha - ((g2*h2)/delta) 
    x2 = (b2*h2)/delta 
    x3 = (h2*k2)/delta 
    
    """
    #c2 p1 
    x4 = a1 - ((k1*k2)/bb) 
    x5 = (k2*epsil)/bb 
    x6 = ((hb2*k2)/bb) 
    """
    """
    #c2 np1 
    x33 = beta - ((k1*k2 + g1*hb)/FF) - ((a2*b1*hb)/(FF*h2)) 
    x4 = a1 - ((a2*hb*alpha)/(FF*h2))
    x5 = ((ab*k2)/FF) + ((a2*hb*k2)/(FF*h2))
    x6 = 0 
    """
    #c2 np2 
    x33 = beta - ((k1*k2 - g1*hb)/FF)
    x4 = a1 - ((a2*g2*hb)/(FF*delta))
    x44 = h1 + ((hb*gamma)/FF) - ((a2*b2*hb)/(FF*delta))
    x5 = (ab*k2)/FF
    x6 = (a2*hb*k2)/(FF*delta) 
    
    #o4 by 3>3, 8>8>2
    x7 = gamma - ((a2*b2)/delta) - ((k1*k2)/G) + ((gb*h1)/G) 
    x8 = ((a2*g2)/delta) - ((a1*gb)/G)
    x9 = g1 + ((gb*beta)/G) 
    x10 = ((a2*k2)/delta) + ((ab2*k2)/G) 
    
    """
    #o4 by 8>7>3>3 
    x77 = a2 - ((k1*k2)/bb)
    x7 = gamma - b2*(x77/delta)
    x8 = g2*(x77/delta)
    x9 = g1 
    x10 = ((H*k2)/bb2) + k2*(x77/delta)
    """
    
    #c5 6>6>8>4 
    x11 = epsil - ((ab*bb)/FF)
    x12 = k1 - ((bb*hb*a2*g2)/delta) 
    x13 = ((k1*bb)/FF) - ((bb*g1*hb)/(FF*k2))
    x14 = ((bb*hb*gamma)/(FF*k2))
    x15 = hb2 - ((bb*hb*a2)/(FF*delta))
    
    """
    #c5 by 6>2 
    x11 = epsil 
    x12 = k1 - ((a1*bb)/k2)
    x13 = (bb*beta)/k2
    x14 = (h1*bb)/k2 
    x15 = hb2 
    """
    
    """
    #o7 by 8>4 
    x16 = H - ((k1*k2 - a2*bb2)/delta) 
    x17 = ((g2*k1*k2 - a2*bb2)/(delta*k2))
    x18 = (g1*bb2)/k2 
    x19 = ((gamma*bb2)/k2) + ((b2*k1*k2 - a2*bb2)/(k2*delta))
    """
    
    #o7 by 8>8>2, 3>3 
    th1 = H - ((ab2*bb2)/G) 
    th2 = (k1*bb2)/G 
    th3 = ((bb2*gb)/(G*k2)) 
    
    x16 = th1 - ((k1*k2)/delta) 
    x17 = ((g2*k1)/delta) - (a1*th3) 
    x18 = th3*beta 
    x19 = th2 - h1*th3 + ((b2*k1)/delta) 
    
    
    #y
    y1 = x1 - x8*(x2/x7) 
    y2 = b1 + x9*(x2/x7) 
    y3 = x3 + x10*(x2/x7) 
    
    """
    #c2 p1 
    #y4 = beta - x9*(h1/x7) 
    
    #c2 np1 
    y4 = x33 - x9*(h1/x7)
    
    y5 = x4 + x8*(h1/x7) 
    y6 = x10*(h1/x7) - x6
    """    
    
    #c2 np2 
    y4 = x33 - x9*(x44/x7)
    y5 = x4 + x8*(x44/x7) 
    y6 = x10*(x44/x7) - x6
    
    y7 = x12 + x8*(x14/x7) 
    y8 = x13 + x9*(x14/x7) 
    y9 = x15 + x10*(x14/x7) 
    
    y10 = x16 - x10*(x19/x7) 
    y11 = x17 + x8*(x19/x7) 
    y12 = x9*(x19/x7) - x18 
    
    #z 
    z1 = y1 - y5*(y2/y4)
    z2 = k2 + x5*(y2/y4)
    z3 = y3 + y6*(y2/y4)
    
    z4 = x11 - x5*(y8/y4)
    z5 = y7 + y5*(y8/y4)
    z6 = y9 + y6*(y8/y4)
    
    z7 = y10 - y6*(y12/y4)
    z8 = y11 + y5*(y12/y4)
    z9 = gb2 + x5*(y12/y4)
    
    #e
    e1 = z7 - z3*(z8/z1) 
    e2 = z9 + z2*(z8/z1) 
    e3 = e2/e1 
    
    e4 = ((z4 - z6*e3)/z5) 
    
    e5 = y5*e4 + x5 + y6*e3 
    e6 = e5/y4 
    
    e7 = x8*e4 + x9*e6 + x10*e3 
    e8 = e7/x7 
    
    e9 = g2*e4 + b2*e8 + k2*e3 
    e10 = e9/delta 
    
    e11 = ((gb/FF)*(ab + k1*e6)) + ab2*e3 + k1*e8 
    e12 = G - ((gb*hb)/FF) 
    e13 = e11/e12 
    
    e14 = ((hb/G)*(ab2*e3 + k1*e8)) + ab + k1*e6 
    e15 = FF - ((gb*hb)/G) 
    e16 = e14/e15 
    
    #equations n8 
    c5_n8 = 1/(1 + e4 + e6 + e10 + e8 + e16 + e3 + e13) 
    c1_n8  = e4*c5_n8 
    c2_n8 = e6*c5_n8 
    o3_n8 = e10*c5_n8 
    o4_n8 = e8*c5_n8 
    c6_n8 = e16*c5_n8 
    #c6_n8 = (1/2) - (1 + e13 + e3)*c5_n8 
    o7_n8 = e3*c5_n8 
    o8_n8 = e13*c5_n8 
    #o8_n8 = (1/2) - (1 + e16 + e3)*c5_n8 
    
    dn8 = {} 
    dn8 = { 'c1' : c1_n8, 'c2' : c2_n8, 'o3' : o3_n8, 'o4' : o4_n8, 'c5' : c5_n8, 'c6' : c6_n8, 'o7' : o7_n8, 'o8' : o8_n8}
    ndn8.update({ vol : dn8 })
    
    SS_n8 = [c1_n8, c2_n8, o3_n8, o4_n8, c5_n8, c6_n8, o7_n8, o8_n8]
    const = np.linalg.solve(V, SS_n8) #obtain constants that scale eigenvectors into predicted steady state values 
    
    #SS_n8 = np.array([[c1_n8, c2_n8, o3_n8, o4_n8, c5_n8, c6_n8, o7_n8, o8_n8]])
    #const = np.linalg.solve(V, SS_n8.T) 
    
    #analytical simulations
    if_l = [] 
    ss_l = {}
    for x in range(0, 410, 10):
        cru = sum(const[i]*np.exp(EigD[i]*x)*V[0,i] for i in range(0, 8))
        cau = sum(const[i]*np.exp(EigD[i]*x)*V[1,i] for i in range(0, 8))
        oru = float(sum(const[i]*np.exp(EigD[i]*x)*V[2,i] for i in range(0, 8)).real)
        oau = float(sum(const[i]*np.exp(EigD[i]*x)*V[3,i] for i in range(0, 8)).real)
        crb = sum(const[i]*np.exp(EigD[i]*x)*V[4,i] for i in range(0, 8))
        cab = sum(const[i]*np.exp(EigD[i]*x)*V[5,i] for i in range(0, 8))
        orb = float(sum(const[i]*np.exp(EigD[i]*x)*V[6,i] for i in range(0, 8)).real)
        oab = float(sum(const[i]*np.exp(EigD[i]*x)*V[7,i] for i in range(0, 8)).real)
        
        #totals
        C_t = cru + cau + crb + cab
        O_t = float((oru + oau + orb + oab).real)
        U_t = cru + cau + oru + oau 
        B_t = crb + cab + orb + oau 
        O_ut = oau + oru 
        O_bt = oab + orb 

        ss_l.update( {x : O_t} ) 

    nss.update( {vol : ss_l} )   
    
last1 = nss[-155][8000]
for k in nss.keys(): 
    nss[k] = {key:(val/last1) for key, val in nss[k].items()}
    
last2 = nss[40][0]
for k in [-10, 10, 40]:
    nss[k] = {key:(val/last2) for key, val in nss[k].items()}
  
dss = pd.DataFrame.from_records(nss, coerce_float=True)

f1 = plt.figure()
f2 = plt.figure() 

xval1 = dfMerge.iloc[:, [0, 2, 4, 6, 8, 10]] #WT activation
yval1 = dfMerge.iloc[:, [1, 3, 5, 7, 9, 11]] #WT activation
xval2 = dfMerge.iloc[:, [12, 14, 16]] #WT deactivation
yval2 = dfMerge.iloc[:, [13, 15, 17]] #WT deactivation

print(dss)

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
