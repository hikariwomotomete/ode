import glob
import os 
from matplotlib import cm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#path to .csv files containing processed data 
WTpath =r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\kim2012\new csv\WT\milliseconds'
FApath =r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\kim2012\new csv\FA\milliesconds'
#path = FApath #specify which files to use, FA or WT in this case 

#separate activation and deactivation files into respective lists 
def file_sep(path):
    act_files = glob.glob(path + "/*act*.csv") 
    de_files = glob.glob(path + "/*de*.csv")
    
    if len(de_files) > 2: 
        myorder = [1, 2, 0] 
    else:
        myorder = [1, 0] 
    
    ordered_de_files = [de_files[i] for i in myorder] 
    return act_files, ordered_de_files 

'''#data files have been named in following format: 
kim2012_[WT or FA]_[act or de]_[voltage]mV_new.csv
e.g. kim2012_WT_act_-120mV_new.csv

to get prefices (voltage value) for each csv file, need to consider following lengths: 
/*_new.csv = 8 characters 
then, voltage will be [-11:-8] for all 2 digit voltages, e.g. +20, -20,
but 3 digit voltages will be [-12:-8], e.g. -120, -100 

but FA files don't have the 'mV,' so subtract two from indices below**
'''

#create function to merge data properly 
def df_merging(file_list, df_list): 
    #file_list is the list that contains the data files 
    #df_list is a list that will contain output dfs 
    
    leg_list = [] 
    
    n = len(file_list) 
    for i in range(n): 
        f = file_list[i] 
        prefix = os.path.basename( os.path.normpath(f) ) #get the filename without directory 
        
        if 'mV' in str(prefix): #then WT 
            i = -13 
            k = -10 
        else: #FA 
            i = -11 
            k = -8
        if '-1' in str(prefix): #3 digit voltages 
            short_prefix = int(prefix[i-1 : k]) #just get the voltage 
        elif '_0' in str(prefix): 
            short_prefix = 0 
        else: #2 digit voltages 
            short_prefix = int(prefix[i : k]) 
        
        leg_list.append(short_prefix) 
        
        #convert file to dataframe, then rename y column header to voltage 
        df = pd.read_csv(f).rename(columns = {'Curve1' : str(short_prefix)} ) 
        
        df_list.append(df) 
    
    return df_list, leg_list 

def df_merger(faorwt):
    if faorwt == 'FA': 
        p = FApath 
    else:
        p = WTpath 
        
    act_files, de_files = file_sep(p) 
    dlist = [] 
    dlist, aleg = df_merging(act_files, dlist) 
    dlist, dleg = df_merging(de_files, dlist)  
    
    return pd.concat(dlist, axis=1, sort=True)
    
#merge data files   
dfMerge_FA = df_merger('FA') 
dfMerge_WT = df_merger('WT') 

def db_pars(params, db): 
    if db == True: 
        a0, sa, b0, sb, d0, sd, g1, h1, g2, h2 = params
        
        new_c0 = (a0*g1*d0*h2)/(b0*h1*g2)
        new_sc = new_sc = 1/( (1/sa) + (1/sb) - (1/sd) )
        new = [a0, sa, b0, sb, new_c0, new_sc, d0, sd, g1, h1, g2, h2]
        
        #print('c0', new_c0) 
        #print('sc', new_sc) 
        
        return new
        
    else:
        return params 

def ss(v, pars, db): 
    a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars(pars, db) 

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

def get_eigs(vol, pars, db):
    a0, sa, b0, sb, c0, sc, d0, sd, g1, h1, g2, h2 = db_pars(pars, db)

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

def get_sim(pars, volstep, volhold, timerange, nssdict, db): 
    vol = volstep 

    EigD, V = get_eigs(vol, pars, db) 
    SS_4s = ss(volhold, pars, db) 
        
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

def full_sim(params, volrange, volhold, timerange, nssdict, db):
    if RepresentsInt(volrange) is True: 
        nssdict = get_sim(params, volrange, volhold, timerange, nssdict, db)
    else:
        for vv in volrange: 
            nssdict = get_sim(params, vv, volhold, timerange, nssdict, db) 
    return nssdict 
   
def fullon_sim(param, db_condition, faorwt): 
    if faorwt == 'FA': 
        de_vols = [-20, 0, 20]
        de_timerange = range(0, 10010, 10)
    else: 
        de_vols = [0, 20] 
        de_timerange = range(0, 810, 10) 
    
    actrange = range(-50, -160, -10)
    derange = range(50, -50, -10)
    
    #in an empty dictionary, fill with simulation 
    nss = {} 
    nss = full_sim(param, actrange, 0, range(0, 16010, 10), nss, db)
    nss = full_sim(param, derange, -100, de_timerange, nss, db)
    
    #normalize 
    act_last = nss[-120][16000] 
    de_last = nss[20][0] 
    for k in range(-50, -130, -10): 
        #open up each nested dict of time:probability tuples in each voltage key nss[k] 
        nss[k] = {key : (value/act_last) for key, value in nss[k].items()}
    for k in de_vols: 
        nss[k] = {key : (value/de_last) for key, value in nss[k].items()}

    #convert into single dataframe 
    #dss = pd.DataFrame.from_dict(nss) 
    
    #return dss 
    
    return nss 

def find_taus(vols, nss, act):
    tau = {} 
    
    if act == True: 
        for k in vols: 
            m = nss[k][16000] 
            m = m/np.exp(1) 
            l1 = []
            l2 = [] 

            for h, j in nss[k].items(): 
                l1.append(h) 
                l2.append( abs(j-m) )
            z = min(l2)
            x = l2.index(z) 
            y = l1[x] 
            tau.update( { k : [y] } )
    else:
        for k in vols: 
            m = nss[k][0] / np.exp(1) 
            l1 = []
            l2 = [] 

            for h, j in nss[k].items(): 
                l1.append(h) 
                l2.append( abs(j- m) )
            z = min(l2)
            x = l2.index(z) 
            y = l1[x] 
            tau.update( { k : [y] } )

    return tau 
    
FA_db = [1.24063886e-01, 1.50000000e+01, 1.55568501e+03, 7.64678684e+01,
       7.00000000e-05, 3.47754083e+01, 2.05565642e-03, 1.48228153e-04,
       8.90728612e-03, 5.07029575e-01]
       
WT_db = [1.66883163e-03, 8.12681740e+00, 1.76662403e+03, 5.40137039e+01,
       8.72393000e-03, 2.10457589e+01, 1.16226019e-03, 6.94138987e-05,
       4.80226109e-05, 9.59073193e-03]

FA_nodb = [0.0281123979, 12.4321539, 1698.87433, 52.691605, 0.0103830274, 7.51102369, 0.820907957, 41.0800331, 2.03431899e-03, 1.00000000e-04, 5.26726209e-06, 2.36210348e-04]

WT_nodb = [0.0634301786, 12.9293755, 1799.99977, 94.9982597, 0.000135395346, 10.473462, 0.00699934753, 37.996743, 0.00287905103, 2.24788874e-05, 2.27056799e-05, 0.0150489481]


#pars_list = [WT_db0, FA_db0, WT_nodb, FA_nodb]
pars_list = [WT_nodb, FA_nodb, WT_db, FA_db]

df_taus = [] 
for i in range(4):   
    param = pars_list[i] 
    
    actrange = range(-50, -160, -10)
    derange = range(50, -50, -10)
    
    if i < 2:
        db = False 
    else:
        db = True 
        
    if (i/2) % 1 == 0:
        fw = 'WT'
    else:
        fw = 'FA' 
    
    print(i, fw)
    nss = fullon_sim(param, db, fw) 
    
    '''
    #normalize 
    last = nss[-150][16000]
    for k in actrange:
        nss[k] = {key : (value/last) for key, value in nss[k].items()}
    last = nss[50][0]
    for k in derange:
        nss[k] = {key : (value/last) for key, value in nss[k].items()}'''

    tau_a = find_taus(actrange, nss, True)
    tau_d = find_taus(derange, nss, False)
    
    ta = pd.DataFrame.from_dict(tau_a) 
    td = pd.DataFrame.from_dict(tau_d) 
    tt = pd.concat([ta, td], axis=1) 
    
    df_taus.append(tt) 
    
    
#find taus for data
def finder(df, faorwt):
    
    #determine number of deactivation columns 
    if faorwt == 'FA':
        r = 22 
    else:
        r = 20 
        
    #split data dataframe 
    xval1 = df.iloc[:, range(0, 16, 2)] #activation time 
    yval1 = df.iloc[:, range(1, 17, 2)] #activation probability 
    xval2 = df.iloc[:, range(16, r, 2)] #deactivation time 
    yval2 = df.iloc[:, range(17, r+1, 2)] #deactivation probability
       
    def tauer(u, v):
        #u is yvals 
        #v is xvals 
        
        hds = u.columns.values.tolist()
        n = len(u.columns)
                
        taus = {} 
        for j in range(n):
            y_v = u.iloc[:, j].values.tolist() 
            x_v = v.iloc[:, j].values.tolist() 
            
            if n < 5:
                m = np.exp(-1) 
            else:
                m = max( y_v ) / np.exp(1)  
                
            diffs = [ abs(x - m) for x in y_v ] 
            tau_v = diffs.index( min(diffs) )
            
            #tau_y_idx = y_v[tau_v]       
            tau = x_v[tau_v]
            
            taus.update({ int(hds[j]) : [tau] }) 
            
        return taus
    
    t_act = tauer(yval1, xval1) 
    t_de = tauer(yval2, xval2) 
    
    return t_act, t_de 
    
    
t_fa_act, t_fa_de = finder(dfMerge_FA, 'FA') 
t_wt_act, t_wt_de = finder(dfMerge_WT, 'WT') 
    
xvals = range(0, 16001, 1) 
heads = list(df_taus[0].columns.values)
print(heads) 
titles = ['WT without detailed balance', 'FA without detailed balance', 'WT with detailed balance', 'FA with detailed balance'] 

print(df_taus[1])
print(df_taus[3])

f = plt.figure() 
ax1 = f.add_subplot(121) 
ax2 = f.add_subplot(122)

#move ax2 y-axis to the right
#ax2.yaxis.tick_right() 

clrs = ['r', 'b', 'r', 'b', 'cyan', 'magenta', 'purple', 'g'] 
legs = [] 

#data 
data_list = [t_wt_act, t_fa_act, t_wt_de, t_fa_de]    

tau_data = [] 
for j in data_list: 
    df = pd.DataFrame.from_dict(j).T.reset_index().sort_values('index', axis=0, ascending=False) 
    tau_data.append(df) 
    
data_titles = ['WT activation', 'FA activation', 'WT deactivation', 'FA deactivation'] 

#legend use 
h1 = [] 
l1 = [] 
h2 = [] 
l2 = [] 

for x in range(4):
    d = df_taus[x] 
    t = titles[x] 
    
    #plot data 
    h = tau_data[x] #data 

    #ls for nodb vs db 
    if x <= 1:
        u = '--'
        curve, = ax1.plot( h.iloc[:,0], h.iloc[:, 1] , color=clrs[x+4], marker='*', markersize=12, linewidth=2, label=data_titles[x] + ' (data)')
                
        h1.append(curve)
        l1.append(curve.get_label())
        
    else:
        u = '-' 
        curve, = ax2.plot( h.iloc[:,0], h.iloc[:, 1] , color=clrs[x+4], marker='*', markersize=12, linewidth=2, label=data_titles[x] + ' (data)') 
        
        h2.append(curve)
        l2.append(curve.get_label())
    
    #dots1 = ax1.scatter(d.T.index[0:11], d.T.iloc[0:11, :], color=clrs[x], marker='o', label=titles[x] + 'activation') 
    #dots1 = ax1.scatter(d.T.index[0:11], d.T.iloc[0:11, :], color=clrs[x], marker='o') 
    #dots2 = ax2.scatter(d.T.index[11:,], d.T.iloc[11:, :], color=clrs[x], marker='^', label=titles[x] + 'deactivation') 
    #dots2 = ax2.scatter(d.T.index[11:,], d.T.iloc[11:, :], color=clrs[x], marker='^') 
    line1, = ax1.plot(d.T.iloc[0:11, :] , color=clrs[x], ls=u, marker='o', markersize=8, label=titles[x] + ' activation') 
    line2, = ax2.plot(d.T.iloc[11:, :], color=clrs[x], ls=u, marker='^', markersize=8, label=titles[x] + ' deactivation') 
    
    h1.append(line1)
    l1.append(line1.get_label())
    h2.append(line2)
    l2.append(line2.get_label())
    
    '''
    handls.append(curve)
    labs.append( data_titles[x] + ' (data)' ) 
    handls.append(line1)
    labs.append(titles[x] + ' activation')
    handls.append(line2) 
    labs.append(titles[x] + ' deactivation') '''
    
    
        
#legend
ax2.legend(h2, l2, fontsize=12)
ax1.legend(h1, l1, fontsize=12)
#plt.legend() 

#create common x label 
#ax_big = f.add_subplot(111, frameon=False)
#ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#ax_big.set_xlabel('Voltage (mV)', fontsize=16, fontweight='bold')

#y label
ax1.set_ylabel(r'$ \tau_{activation} $ (ms)', fontsize=16, fontweight='bold') 
ax2.set_ylabel(r'$ \tau_{deactivation} $ (ms)', fontsize=16, fontweight='bold')

#label
ax1.set_xlabel(r'Voltage (mV)', fontsize=16, fontweight='bold')
ax2.set_xlabel(r'Voltage (mV)', fontsize=16, fontweight='bold')

#titles 
ax1.set_title(r'Activation', fontsize=20, fontweight='bold')
ax2.set_title(r'Deactivation', fontsize=20, fontweight='bold')

#newleg = ax1.add_artist(leg) 
f.suptitle('Time Constants in WT and F431A mHCN2', fontsize=20, fontweight='bold') 

plt.show() 
    