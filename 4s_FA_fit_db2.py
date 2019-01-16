import glob
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from operator import sub 
from scipy.optimize import minimize 

WTpath =r'C:\Users\delbe\Downloads\wut\wut\FALL_2018\Post_grad\accili\2018\lab\digitize\kim2012\new csv\WT'
FApath =r'C:\Users\delbe\Downloads\wut\wut\FALL_2018\Post_grad\accili\2018\lab\digitize\kim2012\new csv\FA'
path = FApath

act_files = glob.glob(path + "/*act*.csv")
de_files = glob.glob(path + "/*de*.csv")

ddact = [] 
aleg = []
for f in act_files: 
    prefix = os.path.basename(os.path.normpath(f)) #get the filename without directory/path 
    if '-1' in str(prefix):
        short_prefix = prefix[-8:-4]
    else: 
        short_prefix = prefix[-7:-4]
    aleg.append(str(short_prefix))
    df = pd.read_csv(f).rename(columns={'Curve1' : str(short_prefix)})
    ddact.append(df) 

dlist = [] 
for da in ddact: 
    df = pd.DataFrame.from_dict(da) 
    dlist.append(df) 

dleg = [] 
for f in de_files:
    prefix = os.path.basename(os.path.normpath(f)) #get the filename without directory/path 
    if 'de_0' in str(prefix):
        short_prefix = 0 #filename
    else: 
        short_prefix = prefix[-7:-4]
    dleg.append(short_prefix)
    df = pd.read_csv(f).rename(columns={'Curve1' : str(short_prefix)})
    dlist.append(df)

dfMerge = pd.concat(dlist, axis=1, sort=True)

def db_pars(params): 
    a0, sa, b0, sb, sc, d0, sd, g1, h1, g2, h2 = params
    new_c0 = (a0*g1*d0*h2)/(b0*h1*g2)
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
    
def get_sim(volstep, volhold, pars, timerange, nssdict, de): 
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

def full_sim(params, merge): 
    x_act = (merge.iloc[:, range(0, 16, 2)].T*1000).values 
    x_de = (merge.iloc[:, range(16, 22, 2)].T*100).values
    nss = {} 
    #nss = get_sim(range(-50, -130, -10), 0, params, range(0, 8000, 10), nss) 
    #nss = get_sim([-20, 0, 20], -100, params, range(0, 500, 10), nss) 
 
    volts = merge.iloc[:, range(1, 23, 2)].columns.values.tolist() 
    for i in range(len(volts)):
        if i < 8:
            nss = get_sim(int(volts[i]), 0, params, x_act[i], nss, False) 
        elif i > 7:
            nss = get_sim(int(volts[i]), -100, params, x_de[i-8], nss, True) 
    
    #last1 = 0.887966831819631
    last = {} 
    last1 = get_sim(-120, 0, params, 8000, last, False)[-120][8000]
    for k in range(-50, -130, -10): 
        nss[k] = {key:(val/last1) for key, val in nss[k].items()}
    #last2 = 0.8299762391569746
    last2 = get_sim(20, -100, params, 0, last, True)[20][0]
    for k in [-20, 0, 20]:
        nss[k] = {key:(val/last2) for key, val in nss[k].items()}
        
     #make sim df 
    df = pd.DataFrame.from_dict(nss) 
    return df

def cost(params): 
    dss = full_sim(params, dfMerge) 
    merge_y = dfMerge.iloc[:, range(1, 23, 2)]
    
    dif = [] 
    sims = [] 
    longur = []
    
    for x in range(0, 8, 1):
        sim = [i for i in dss.iloc[:, x].tolist() if RepresentsInt(i) is True]
        data = [i for i in merge_y.iloc[:, x].tolist() if RepresentsInt(i) is True]
        #sims.append(sim) 
        longur.append(len(sim))
        
        error = [ ( ( ( sim[j] - data[j] )**2) / (data[j]**2) ) for j in range( len(sim) )]
        #error = [ (abs( data[j] - sim[j] ) / ( abs( sim[j] ) + abs( data[j] ) ) )  for j in range( len(sim) )]
          
        dif.append(sum(error))
        
    '''for o in range( len(longur) ):
        longur[o] = longur[o] / sum(longur) 
        dif[o] = dif[o] * longur[o]  '''
        
    diff = sum(dif)
    print(sum(dif[0:9])) 
    print(sum(dif[8:11])) 
    return diff 
#
#pars_FA = [0.03, 11.5, 1500, 80, 11, 0.9, 40, 0.0028, 0.0002, 5e-05, 0.005]
pars_FA = [3.34996804e-02, 1.14738452e+01, 1.50004791e+03, 8.00724765e+01,
       1.09554981e+01, 9.48848497e-01, 4.00643486e+01, 2.75000000e-03,
       3.76428550e-04, 4.98850138e-05, 4.96651118e-03]
#
bnds3 = ((1e-3, 50e-3), (8, 13), (500, 2000), (40, 200), (10, 11), (0.9, 1), (30, 200), (25e-4, 50e-4), (10e-5, 25e-5), (4.5e-5, 5e-5), (45e-4, 55e-4))
#
bnds4 = [(i/5, 5*i) for i in pars_FA]
##
##
param = pars_FA
par_label = 'pars_FA'
##
##
res = minimize(cost, param, method='L-BFGS-B', bounds=bnds4)
pars_iter = res['x'] 
dss = full_sim(pars_iter, dfMerge) 
print(res)
print(cost(param), cost(pars_iter))
#
#
#make two separate figures 
f1 = plt.figure()

#get data from digitized leo thesis 
xval1 = dfMerge.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]]*1000 #WT activation
yval1 = dfMerge.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]] #WT activation
xval2 = dfMerge.iloc[:, [16, 18, 20]]*100 #WT deactivation
yval2 = dfMerge.iloc[:, [17, 19, 21]] #WT deactivation

de_x1 = xval2.iloc[:, 0]
de_x2 = xval2.iloc[:, 1]
de_x3 = xval2.iloc[:, 2]
de_y1 = yval2.iloc[:, 0]
de_y2 = yval2.iloc[:, 1]
de_y3 = yval2.iloc[:, 2]

#get de/activation data to plot (model) 
act_sim = dss.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
de_sim = dss.iloc[:, [8, 9, 10]]

#get column headings to use as legend
volts = dss.columns.values.tolist() 
#
font = {'family' : 'normal',
        'size'   : 16}
#
plt.rc('font', **font)
#activation plots 
ax1 = f1.add_subplot(111) 
l1 = ax1.plot(xval1, yval1, ls='--', c='k')
l2 = ax1.plot(act_sim, linewidth=2)
#
ax1.legend(l1, aleg, bbox_to_anchor=(1.11, 1), loc=1, borderaxespad=0.)
sim_act_leg = ax1.legend(l2, volts[0:8], bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0.)
leg3 = ax1.add_artist(sim_act_leg)
#
#deactivation plots 
f2, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
axs[0].plot(de_x1, de_y1, ls='--', c='k')
axs[0].plot(de_sim.iloc[:, 2], linewidth=2)
axs[0].set_title('-20mV')
axs[1].plot(de_x2, de_y2, ls='--', c='k')
axs[1].plot(de_sim.iloc[:, 0], linewidth=2)
axs[1].set_title('0mV')
axs[2].plot(de_x3, de_y3, ls='--', c='k')
axs[2].plot(de_sim.iloc[:, 1], linewidth=2)
axs[2].set_title('+20mV')
#
'''
#deactivation legends
ax3.legend(l3, dleg, loc=1, borderaxespad=0.)
sim_de_leg = ax3.legend(l4, volts[8:11], loc=1, borderaxespad=0.)
leg4 = ax3.add_artist(sim_de_leg) 
'''
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
