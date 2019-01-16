import glob
import os 
from matplotlib import cm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

WTpath =r'C:\Users\delbe\Downloads\wut\wut\FALL_2018\Post_grad\accili\2018\lab\digitize\kim2012\new csv\WT' #path to excel files (WT)
FApath =r'C:\Users\delbe\Downloads\wut\wut\FALL_2018\Post_grad\accili\2018\lab\digitize\kim2012\new csv\FA' #path to excel files (FA) 
path = FApath

act_files = glob.glob(path + "/*act*.csv")
de_files = glob.glob(path + "/*de*.csv")

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
    short_prefix = prefix[-7:-3] #filename
    dleg.append(short_prefix)
    df = pd.read_csv(f) 
    dlist.append(df)
    
dfMerge = pd.concat(dlist, axis=1, sort=True)
print(dleg) 

a0=30e-3
sa=11.5
b0=1500 
sb=80

sc=11 #sa' 
d0=0.9 #b0' 
sd=40 #sb' 

g1=28e-4
h1=20e-5
g2=50e-6
h2=50e-4

pars_FA = [a0, sa, b0, sb, sc, d0, sd, g1, h1, g2, h2] 
pars_FA = [3.34996804e-02, 1.14738452e+01, 1.50004791e+03, 8.00724765e+01,
       1.09554981e+01, 9.48848497e-01, 4.00643486e+01, 2.75000000e-03,
       3.76428550e-04, 4.98850138e-05, 4.96651118e-03]
print(pars_FA)

nss = {} 

def db_pars(params): 
    a0, sa, b0, sb, sc, d0, sd, g1, h1, g2, h2 = params
    new_c0 = (a0*g1*d0*h2)/(b0*h1*g2)
    print('c0', new_c0)
    new = [a0, sa, b0, sb, new_c0, sc, d0, sd, g1, h1, g2, h2]
    return new 
    
def db_pars_de(params):
    a0, sa, b0, sb, sc, d0, sd, g1, h1, g2, h2 = params
    new_c0 = ( (d0*a0*g1*d0*h2)*np.exp(-100/sb) ) / ( b0*h1*g2 )
    print('c0', new_c0)
    new = [a0, sa, b0, sb, new_c0, sc, d0, sd, g1, h1, g2, h2]
    return new 
    
def ss(v, pars, de): 
    if de == True: 
        a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars_de(pars) 
    else: 
        a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars(pars) 

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
        
    #d4s = {} 
    #d4s = { 'c1' : c1_4s, 'c2' : c2_4s, 'o3' : o3_4s, 'o4' : o4_4s}
    SS_4s = [c1_4s, c2_4s, o3_4s, o4_4s]
    
    return(SS_4s) 

def get_eigs(vol, pars, de):
    if de == True: 
        a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars_de(pars) 
    else: 
        a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars(pars) 

    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2 = d0*np.exp(vol/sd)

    #transition rate matrix 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )

    W, V = np.linalg.eig(Q) 
    EigD = [w for w in W]
    
    return EigD, V 

def odes(time, const, eval, evec):

    x = time 
    cr4s = sum(const[i] * np.exp(eval[i]*x) * evec[0,i] for i in range(0, 4))
    ca4s = sum(const[i] * np.exp(eval[i]*x) * evec[1,i] for i in range(0, 4))
    or4s = sum(const[i] * np.exp(eval[i]*x) * evec[2,i] for i in range(0, 4))
    oa4s = sum(const[i] * np.exp(eval[i]*x) * evec[3,i] for i in range(0, 4))

    #totals
    C_t = cr4s + ca4s 
    O_t = or4s + oa4s 
    states = [cr4s, ca4s, or4s, oa4s]
    
    return states

def RepresentsInt(s):
    try:
        int(s)
        return True
    except:
        return False 

def get_sim(pars, volstep, volhold, timerange, nssdict, de): 
    vol = volstep 
    
    if de == True:
        EigD, V = get_eigs(vol, pars, True) 
        SS_4s = ss(volhold, pars, True)     
    else: 
        EigD, V = get_eigs(vol, pars, False) 
        SS_4s = ss(volhold, pars, False) 
        
    const = np.linalg.solve(V, SS_4s) 
    ss_l = {} 
    
    if RepresentsInt(timerange) is True: 
        states = odes(timerange, const, EigD, V) 
        C_t = states[0] + states[1]  
        O_t = states[2] + states[3] 
        ss_l.update( {timerange : O_t} ) 
    else: 
        for x in timerange: 
            states = odes(x, const, EigD, V) 
            C_t = states[0] + states[1]  
            O_t = states[2] + states[3] 
            ss_l.update( {x : O_t} ) 
    nssdict.update({vol : ss_l})
    return nssdict 

def full_sim(params, volrange, volhold, timerange, nssdict, de):
    if RepresentsInt(volrange) is True: 
        nssdict = get_sim(params, volrange, volhold, timerange, nssdict, de)
    else:
        for vv in volrange: 
            nssdict = get_sim(params, vv, volhold, timerange, nssdict, de) 
    return nssdict 
   
##
nss = {} 
##
##
db = None
param = pars_FA
par_label = 'pars_FA'
##
##
nss = full_sim(param, range(-50, -130, -10), 0, range(0, 8010, 10), nss, False)
nss = full_sim(param, [-20, 0, 20], -100, range(0, 510, 10), nss, False)

#normalize 
last = nss[-120][8000]
for k in range(-50, -130, -10):
    nss[k] = {key : (value/last) for key, value in nss[k].items()}
last = nss[20][0]
for k in [-20, 0, 20]:
    nss[k] = {key : (value/last) for key, value in nss[k].items()}
##
dss = pd.DataFrame.from_dict(nss)
##
#make two separate figures 
f1 = plt.figure()

#get data from digitized leo thesis 
xval1 = dfMerge.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]]*1000 #WT activation
yval1 = dfMerge.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]] #WT activation
xval2 = dfMerge.iloc[:, [16, 18, 20]]*100 #WT deactivation
yval2 = dfMerge.iloc[:, [17, 19, 21]] #WT deactivation

de_x1 = xval2.iloc[:, 1] 
de_x2 = xval2.iloc[:, 2] 
de_x3 = xval2.iloc[:, 0] 
de_y1 = yval2.iloc[:, 1]
de_y2 = yval2.iloc[:, 2]
de_y3 = yval2.iloc[:, 0]

#get de/activation data to plot (model) 
act_sim = dss.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
de_sim = dss.iloc[:, [8, 9, 10]]

#get column headings to use as legend
volts = dss.columns.values.tolist() 
#
print(dss) 
#
font = {'family' : 'normal',
        'size'   : 16}
#
plt.rc('font', **font)
#activation plots 
ax1 = f1.add_subplot(111) 
l1 = ax1.scatter(xval1, yval1, c='k')
l2 = ax1.plot(act_sim, linewidth=3)
#
#ax1.legend(l2, aleg, bbox_to_anchor=(1.11, 1), loc=1, borderaxespad=0.)
sim_act_leg = ax1.legend(l2, volts[0:8], bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0.)
leg3 = ax1.add_artist(sim_act_leg)
#
#deactivation plots 
f2, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
axs[0].scatter(de_x1, de_y1, c='k')
axs[0].plot(de_sim.iloc[:, 0], linewidth=3)
axs[0].set_title('-20mV')
axs[1].scatter(de_x2, de_y2, c='k')
axs[1].plot(de_sim.iloc[:, 1], linewidth=3)
axs[1].set_title('0mV')
axs[2].scatter(de_x3, de_y3, c='k')
axs[2].plot(de_sim.iloc[:, 2], linewidth=3)
axs[2].set_title('+20mV')
#
### deactivation 
f3 = plt.figure() 
ax3 = f3.add_subplot(111)
#
l5 = ax3.scatter(de_x1, de_y1, c='r')
l6 = ax3.plot(de_sim.iloc[:, 0], c='r', linewidth=3)
l7 = ax3.scatter(de_x2, de_y2, c='k')
l8 = ax3.plot(de_sim.iloc[:, 1], c='k', linewidth=3)
l9 = ax3.scatter(de_x3, de_y3, c='c')
l10 = ax3.plot(de_sim.iloc[:, 2], c='c', linewidth=3)
#
ax3.text(496, 0.02, '+20mV', fontsize=14)
ax3.text(434, 0.05, '0mV', fontsize=14)
ax3.text(350, 0.065, '+20mV', fontsize=14)
#
#deactivation legends 
ax3.legend([l5, l7, l9], ['-20mV', '0mV', '20mV'], bbox_to_anchor=(1.13, 1), loc=1, borderaxespad=0., fontsize=16)
#
font = {'family' : 'normal',
        'size'   : 22}
#
plt.rc('font', **font)
#
axs[1].set_xlabel('Time (ms)')
ax1.set_xlabel('Time (ms)')
axs[0].set_ylabel('Normalized open fraction')
ax1.set_ylabel('Normalized open fraction')
#
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)
ax1.set_title("F431A activation (%s)" % (par_label), y=1.02)
#ax3.set_title("F431A deactivation (%s, %s)" % (db, par_label), y=1.02)
f2.suptitle("F431A deactivation (%s)" % (par_label), y=0.98)
#
#ax1.legend(aleg)
#ax2.legend(dleg)
ax1.margins(0, 0)
'''
for i in range(3):
    axs[i].margins(0, 0)
'''
plt.show() 

