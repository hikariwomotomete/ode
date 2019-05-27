import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math 
import glob
import os
from scipy.optimize import minimize 

#open csv containing digitized data from moroni's paper (fig. 6D) 
path = r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\2019\moroni_acs' 
data = glob.glob(path + "/*dose_response*.csv") 

#create dataframe containing the experimental data 
dlist = []
labels = []
for d in data: 
    prefix = os.path.basename( os.path.normpath(d) ) 
    labels.append(prefix[14:-4]) 
    
    df = pd.read_csv(d).rename( columns={ 'x' : str(prefix[14:-4]), 'Curve1' : 'Shift' } )
    dlist.append(df) 
    
dfmerge = pd.concat( dlist, axis=1 ) 

#create simulation 
def sim(A, params): 
    #parameters for gating and binding
    #kd1 is the first binding, kd2 the low affinity binding
    #k_i, j 
    #where, i = number of ligands bound, j = opening or closing
    #j = 1 is opening, j=2 is closing 
    kd1, kd2, k11, k12, k21, k22, k31, k32, k41, k42 = params 
    
    #constants
    q = 5
    k0 = 1 
    kd3 = kd2 
    kd4 = kd2 
    
    j_cl = [1, ((4 * A) / (kd1)), ( (6*(A**2)) / (kd1*kd2) ), ( (4*(A**3)) / (kd1*kd2*kd3) ), ( (A**4)/(kd1*kd2*kd3*kd4) )] 
    g_c = [k0, k11, k21, k31, k41] 
    g_o = [k0, k12, k22, k32, k42] 
    
    l_o = (1/sum(j_cl))*sum([ a*b for (a, b) in zip(g_c, j_cl) ])
    l_c = (1/sum(j_cl))*sum([ a*b for (a, b) in zip(g_o, j_cl) ])
    
    dv = 5 * math.log(l_o / l_c) 
    return dv  
    
def full_sim(data, params): 
    
    sim_df_l = [] 
    for n in [0, 2]: 
        data_x = data.iloc[:, n].values.tolist()
        sim_d = {} 

        #evaluate model at each ligand concentration 
        #concentration and model value are in key-value of a dict
        for c in data_x: 
            if c != np.inf: 
                sim_d.update( { c : [sim(c, params)] } ) 

        #convert dict to df 
        sim_df = pd.DataFrame.from_dict( sim_d, orient='index' ).reset_index()  

        #add dict of each ligand to a list of dfs 
        sim_df_l.append( sim_df ) 
    
    sim_merge = pd.concat(sim_df_l, axis=1)    
    return sim_merge 
    
def rmse(y_true, y_pred): 
    n = len(y_pred)
    error_sum = 0
    
    for i in range(n):
        if y_pred[i] != 0: 
            error1 = (y_true[i] - y_pred[i]) / y_pred[i]
            error2 = np.square(error1)
            error_sum += error2 
    error3 = (error_sum)/n
    error4 = np.sqrt(error3)
    
    return error4

def cost(params, n): 
    #n determines whether function is for WT or 7CH     
    #0 = 7CH, 2 = cAMP 
    
    #not needed because only evaluating error for one curve at a time 
    #dif = [] 
    data = dfmerge 
    
    data_x = [x for x in data.iloc[:, n].values.tolist() if not np.isnan(x)]
    data_y = [y for y in data.iloc[:, n+1].values.tolist() if not np.isnan(y)] 
    
    sim_y = [sim(c, params) for c in data_x] 
    
    error = rmse( data_y, sim_y ) 
    
    return error 
    
#WT pars 
#kd1, kd2, k11, k12, k21, k22, k31, k32, k41, k42
pars_WT = [0.09, 1.39, 1.6,
        1.01, 10, 4, 
        80, 38, 120, 
        40] 
pars_7CH = [4.47974054e-01, 2.20256601e+01, 8.83008989e+00, 
            1.10134488e+01, 4.36364700e+03, 6.70177700e+01,
            6.13655823e+00, 1.40944143e+02, 3.01112805e+02, 
            9.30481848e+01]
pars_cAMP = [1.76193516e-02, 1.17766075e-01, 5.22910519e-08, 
            2.12542082e+00, 2.35373824e-03, 2.64721576e-02,
            7.05386408e+01, 4.79825409e+01, 1.31817598e+02, 
            2.81648105e+00]

#optimize 
param = pars_WT 
n = 2 
u = 1
bds = [ (1e-9, u), (1, u*10), (1, 5), (1, 5), 
        (5, 20), (1, 5), (50, 100), (1, 50),
        (100, 1000), (1, 100) ]
        
res = minimize(cost, param, args=(n), method='L-BFGS-B', bounds=bds)
pars_iter = res['x'] 
print(res) 
print(cost(param, n), cost(pars_iter, n)) 

#plot
n_labels = ['7CH', 0, 'cAMP']
pars_7CH_new =  [6.75e-9, 5e-8, 6.97033924e-03, 2.85503835e-01,
       8.64879636e+00, 3.06135551e+00, 5.58298498e+01, 3.06756766e+01,
       5.90451733e+02, 1.00000000e+01]
params = [pars_7CH_new, 0, pars_cAMP] 
params = [pars_7CH_new, 0, pars_iter] 


for i in [0, 2]: 
    x_data = dfmerge.iloc[:, i].values.tolist()  
    x_sim = np.linspace( min(x_data), max(x_data), 1e4 ) 
    
    y_sim_original = [sim(c, params[i]) for c in x_sim] 
    y_sim_optimized = [sim(c, params[i]) for c in x_sim] 
    
    plt.plot( x_sim, y_sim_original, ls='--', c='r', label='original' + n_labels[i] ) 
    plt.plot( x_sim, y_sim_optimized, ls='--', c='b', label='optimized' + n_labels[i] ) 
    
    plt.plot( x_data, dfmerge.iloc[:, i+1], label='data' + n_labels[i]) 
    plt.scatter( x_data, dfmerge.iloc[:, i+1], label='data' + n_labels[i]) 
 
plt.legend() 
plt.xscale('log') 
#plt.show() 

#yxli 
pars_WT = [0.09, 1.39, 1.6,
        1.6/2, 10, 10/4, 
        80, 80/78, 120, 
        120/40] 
pars_WT = [1.01, 4, 38, 40] 
        
path = r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\digitize\2019\yxli_steadystate_campshift' 
data = glob.glob( path + '/*.csv' ) 

dfl = [] 
for d in data: 
    prefix = os.path.basename( os.path.normpath(d) ) 
    df = pd.read_csv( d ).rename( columns={ 'Curve1' : str(prefix[0:-4])} ) 
    dfl.append(df) 
dfmerge = pd.concat( dfl, axis=1 ) 
print(dfmerge) 
    
def sim(A, params, n): 
    #parameters for gating and binding
    #kd1 is the first binding, kd2 the low affinity binding
    #k_i, j 
    #where, i = number of ligands bound, j = opening or closing
    #j = 1 is opening, j=2 is closing 
    kd1, kd2, k11, k21, k31, k41 = [0.09, 1.39, 1.6, 10, 80, 120] 
    
    a, b, c, d = params 
    
    #constants
    
    k12 = k11/a 
    k22 = k21/b 
    k32 = k31/c 
    k42 = k41/d 
    
    kd3 = kd2 
    kd4 = kd2 
    
    if n == 4: 
    
        j_cl = [1, ((4 * A) / (kd1)), ( (6*(A**2)) / (kd1*kd2) ), ( (4*(A**3)) / (kd1*kd2*kd3) ), ( (A**4)/(kd1*kd2*kd3*kd4) )] 
        g_c = [1, k11, k21, k31, k41] 
        g_o = [1, k12, k22, k32, k42] 
    
    if n == 3: 
        
        j_cl = [1, ((3 * A) / (kd1)), ( (3*(A**2)) / (kd1*kd2) ), ( (1*(A**3)) / (kd1*kd2*kd3) ) ] 
        g_c = [1, k11, k21, k31] 
        g_o = [1, k12, k22, k32]
        
    if n == 2: 
        
        j_cl = [1, ((2 * A) / (kd1)), ( (1*(A**2)) / (kd1*kd2) ) ] 
        g_c = [1, k11, k21] 
        g_o = [1, k12, k22]
        
    if n == 1: 
        
        j_cl = [1, ((1 * A) / (kd1))] 
        g_c = [1, k11] 
        g_o = [1, k12]        
    
    l_o = (1/sum(j_cl))*sum([ a*b for (a, b) in zip(g_c, j_cl) ])
    l_c = (1/sum(j_cl))*sum([ a*b for (a, b) in zip(g_o, j_cl) ])        
    
    dv = 5 * math.log(l_o / l_c) 
    return dv  
    
def cost( params ): 
    
    dif = [] 
    for i in range(0, 8, 2): 
        head = dfmerge.columns.values.tolist()[i + 1]
        x_data = dfmerge.iloc[:, i].values.tolist() 
        y_data = dfmerge.iloc[:, i+1].values.tolist() 
        #x_sim = np.linspace( min(x_data), max(x_data), 1e4 )
        y_sim = [sim(x, params, ( (i/2) + 1 ) ) for x in x_data] 
        
        error = rmse(y_data, y_sim) 
        dif.append(error) 
        
    return sum(dif) 
    
bds = [(1, None), (1, None), (1, None), (1, None)] 
res = minimize( cost, pars_WT, method='L-BFGS-B', bounds=bds ) 
pars_iter = res['x']
print(res) 
    
f = plt.figure() 
ax = f.add_subplot(111) 
for i in range(0, 8, 2): 
    head = dfmerge.columns.values.tolist()[i + 1]
    x_data = dfmerge.iloc[:, i].values.tolist() 
    y_data = dfmerge.iloc[:, i+1].values.tolist() 
    x_sim = np.linspace( min(x_data), max(x_data), 1e4 )
    y_sim = [sim(x, pars_iter, ( (i/2) + 1 ) ) for x in x_sim] 
    
    ax.plot(x_sim, y_sim, ls='--', label=str(head) + ' reproduced') 
    ax.scatter(x_data, y_data, label=str(head) + ' yxli') 
    
plt.legend() 
plt.xscale('log') 
plt.show() 