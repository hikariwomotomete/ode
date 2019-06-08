import glob
import os 
from matplotlib import cm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 

#path to .csv files containing processed data 
WTpath =r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\kim2012\new csv\WT\milliseconds'
FApath =r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\kim2012\new csv\FA\milliesconds'
path = FApath #specify which files to use, FA or WT in this case 

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
            short_prefix = prefix[i-1 : k] #just get the voltage 
        else: #2 digit voltages 
            short_prefix = prefix[i : k] 
        
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
    
    #in an empty dictionary, fill with simulation 
    nss = {} 
    nss = full_sim(param, range(-50, -130, -10), 0, range(0, 16010, 10), nss, db)
    nss = full_sim(param, de_vols, -100, de_timerange, nss, db)
    
    #normalize 
    act_last = nss[-120][16000] 
    de_last = nss[20][0] 
    for k in range(-50, -130, -10): 
        #open up each nested dict of time:probability tuples in each voltage key nss[k] 
        nss[k] = {key : (value/act_last) for key, value in nss[k].items()}
    for k in de_vols: 
        nss[k] = {key : (value/de_last) for key, value in nss[k].items()}

    #convert into single dataframe 
    dss = pd.DataFrame.from_dict(nss) 
    
    return dss 

def plotter(data_df, sim_df, faorwt): 
    if faorwt == 'FA': 
        j = 22
        k = 3 
    else:
        j = 20
        k = 2
    
    #split data dataframe 
    xval1 = data_df.iloc[:, range(0, 16, 2)] #activation time 
    yval1 = data_df.iloc[:, range(1, 17, 2)] #activation probability 
    xval2 = data_df.iloc[:, range(16, j, 2)] #deactivation time 
    yval2 = data_df.iloc[:, range(17, j+1, 2)] #deactivation probability
    
    print(yval2)
    
    #split deactivation 
    if j > 20: 
        de_sim = sim_df.iloc[:, [8, 9, 10]]
    else:
        de_sim = sim_df.iloc[:, [8, 9]]
    
    act_sim = sim_df.iloc[:, range(0, 8, 1)]
    
    #make figures 
    f_act = plt.figure() #activation kinetics 
    f_de1 = plt.figure() #deactivation 
        
    #column headings to use as legend 
    volts = sim_df.columns.values.tolist() 
    print(volts)
    
    #colour list 
    clrs = ['y', 'b', 'r', 'g', 'c', 'magenta', 'salmon', 'lightgreen'] 
    
    #activation plot 
    ax1 = f_act.add_subplot(111) 
    ax1_l1 = ax1.plot(xval1, yval1, c='k', linewidth = 2) 
    ax1_l2 = ax1.plot(act_sim, linewidth = 4, ls='--') 
    
    #activation legend 
    sim_act_leg = ax1.legend(ax1_l2, volts[0:8], bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0., fontsize=12)
    ax1_leg = ax1.add_artist(sim_act_leg) 
    
    #separate deactivation plots 
    if j > 20: 
        f2, axs = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True) 
        axs[0].scatter(xval2.iloc[:, 0], yval2.iloc[:, 0], c='k')
        axs[0].plot(de_sim.iloc[:, 0], linewidth=3)
        axs[0].set_title('-20mV', fontsize=16)
        
        axs[1].scatter(xval2.iloc[:, 1], yval2.iloc[:, 1], c='k')
        axs[1].plot(de_sim.iloc[:, 1], linewidth=3)
        axs[1].set_title('0mV', fontsize=16)
        
        axs[2].scatter(xval2.iloc[:, 2], yval2.iloc[:, 2], c='k', label='Data')
        axs[2].plot(de_sim.iloc[:, 2], linewidth=3, label='Model')
        axs[2].set_title('+20mV', fontsize=16)
        
        #deactivation legend 
        axs[2].legend(bbox_to_anchor=(1.22, 1), loc=1, fontsize=16)
    else: 
        f2, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        axs[0].scatter(xval2.iloc[:, 0], yval2.iloc[:, 0], c='k')
        axs[0].plot(de_sim.iloc[:, 0], linewidth=3)
        axs[0].set_title('0mV', fontsize=16)

        axs[1].scatter(xval2.iloc[:, 1], yval2.iloc[:, 1], c='k', label='Data')
        axs[1].plot(de_sim.iloc[:, 1], linewidth=3, label='Model')
        axs[1].set_title('20mV', fontsize=16)

        #deactivation legend 
        axs[1].legend(bbox_to_anchor=(1.22, 1), loc=1, fontsize=16)
        
    axs[0].set_ylabel('Normalized Open Fraction', fontsize=16, fontweight='bold') 
    
    #create common x label 
    ax_big = f2.add_subplot(111, frameon=False)
    ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_big.set_xlabel('Time (ms)', fontsize=16, fontweight='bold') 
    
    f2.suptitle("%s deactivation (%s)" % (faorwt, par_label), y=0.98, fontsize=18, fontweight='bold')
    
    '''#make big subplot for whole axes labels 
    ax_de2 = f2.add_subplot(111, zorder=3)
    
    # Turn off axis lines and ticks of the big subplot
    ax_de2.spines['top'].set_color('none')
    ax_de2.spines['bottom'].set_color('none')
    ax_de2.spines['left'].set_color('none')
    ax_de2.spines['right'].set_color('none')
    ax_de2.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    
    ax_de2.set_xlabel('Time (ms)', fontsize=18) 
    ax_de2.set_ylabel('Normalized Open Probability', fontsize=18) '''
    
    #deactivation 
    ax2 = f_de1.add_subplot(111) 

    l_sim = ax2.plot(de_sim.iloc[:, range(k)], ls='--', linewidth=2) 
    l_data = ax2.plot(xval2.iloc[:, range(k)], yval2.iloc[:, range(k)], linewidth=2)
    
    for p in [l_sim, l_data]:
        for line in p: 
            n = int( p.index( line ) )
            line.set_color( clrs[n] )  
    
    #deactivation legend 
    sim_de_leg = ax2.legend(l_sim, volts[8:12], bbox_to_anchor=(1.08, 1), loc=1, borderaxespad=0., fontsize=12) 
    ax2_leg = ax2.add_artist(sim_de_leg) 
    
    ax2.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Normalized Open Fraction', fontsize=16, fontweight='bold') 
    ax1.set_ylabel('Normalized Open Fraction', fontsize=16, fontweight='bold') 
    
    ax1.set_title('%s activation (%s)' % (faorwt, par_label), y=1.02, fontsize=18, fontweight='bold')
    ax2.set_title('%s deactivation (%s)' % (faorwt, par_label), y=1.02, fontsize=18, fontweight='bold')

'''pars_FA = [2.80923471e-02, 1.24320627e+01, 1.69887433e+03, 5.26915979e+01,
       1.03830677e-02, 7.51102343e+00, 8.20904121e-01, 4.10800323e+01,
       1.62092700e-03, 8.75707038e-05, 6.02985683e-06, 2.37048546e-04]
pars_WT = [0.0634301786, 12.9293755, 1799.99977, 94.9982597, 0.000135395346, 10.473462, 0.00699934753, 37.996743, 0.00287905103, 2.24788874e-05, 2.27056799e-05, 0.0150489481] '''

#new db 
pars_FA = [1.24063886e-01, 1.50000000e+01, 1.55568501e+03, 7.64678684e+01,
       7.00000000e-05, 3.47754083e+01, 2.05565642e-03, 1.48228153e-04,
       8.90728612e-03, 5.07029575e-01]
       
pars_WT = [1.66883163e-03, 8.12681740e+00, 1.76662403e+03, 5.40137039e+01,
       8.72393000e-03, 2.10457589e+01, 1.16226019e-03, 6.94138987e-05,
       4.80226109e-05, 9.59073193e-03]
       
db = True
if db == True: 
    par_label = 'with detail balance'
else: 
    par_label = 'no db'
    
dss_WT = fullon_sim(pars_WT, db, 'WT') 
dss_FA = fullon_sim(pars_FA, db, 'FA') 

#dss_WT.to_excel('wt_nodb.xlsx')
#dss_FA.to_excel('fa_nodb.xlsx') 

plotter(dfMerge_WT, dss_WT, 'WT') 
plotter(dfMerge_FA, dss_FA, 'FA') 

#dfMerge_WT.to_excel('wt_data.xlsx') 
#dfMerge_FA.to_excel('fa_data.xlsx')

plt.show() 

def cost(params, db): 
    dss = fullon_sim(params, db, 'WT') 
    merge_y = dfMerge_WT.iloc[:, range(1, 21, 2)]
    
    heads = merge_y.columns.values.tolist()
    
    ddif = {} 
    dif = [] 
    sims = [] 
    longur = []
    
    def rmse(y_true, y_pred): 
        n = len(y_true)
        error_sum = 0
        for i in range(n):
            error1 = (y_true[i] - y_pred[i]) / y_pred[i]
            error2 = np.square(error1)
            error_sum += error2 
        error3 = (error_sum)/n
        error4 = np.sqrt(error3)
        return error4 
    
    for x in range(0, 10, 1):
        sim = [i for i in dss.iloc[:, x].tolist() if RepresentsInt(i) is True]
        data = [i for i in merge_y.iloc[:, x].tolist() if RepresentsInt(i) is True]
        longur.append(len(sim))
        
        error = rmse(data, sim) 
     
        dif.append(error)
        ddif.update( {heads[x] : error} ) 
        
    diff = sum(dif)
    
    return diff

''' res = minimize(cost, pars_WT, args=(db), method='L-BFGS-B') 
print(res)
pars_iter = res['x']
print(cost(pars_WT, db), cost(pars_iter, db)) 
dss_WT = fullon_sim(pars_iter, db, 'WT') 
plotter(dfMerge_WT, dss_WT, 'WT') 
plt.show() '''