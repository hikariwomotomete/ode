import os
import glob
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

path = r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\kinetic_models\four\parameterization\newdb_xpp'

fa_path = glob.glob(path + '/fa/*.dat') 
wt_path = glob.glob(path + '/wt/*.dat') 

def get_xpp(path): 
    
    df_list = [] 
    
    for n in range(len(path)): 
        df = pd.read_csv( path[n], sep="\s+", header = None ) 
        df = df.rename( columns = { 5 : str( df.iloc[0, 7] ) } )
        df = df.iloc[:, [0, 5]]
        
        df_list.append(df) 
        
    dfmerge = pd.concat(df_list, axis=1) 
    return dfmerge 

fa_xpp = get_xpp(fa_path) 
wt_xpp = get_xpp(wt_path) 

sim_path = r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\kinetic_models\four\parameterization'
wt_sim = sim_path + '/wt_newdb.xlsx' 
fa_sim = sim_path + '/fa_newdb.xlsx' 

wt_sim = pd.read_excel(wt_sim) 
fa_sim = pd.read_excel(fa_sim)

wt_data = pd.read_excel( sim_path + '/wt_data.xlsx' ) 
fa_data = pd.read_excel( sim_path + '/fa_data.xlsx' ) 

wt_data_x = wt_data.iloc[:, range(0, 20, 2)] 
fa_data_x = fa_data.iloc[:, range(0, 22, 2)] 
wt_data_y = wt_data.iloc[:, range(1, 21, 2)] 
fa_data_y = fa_data.iloc[:, range(1, 23, 2)] 

fa_xpp_x = fa_xpp.iloc[:, range(0, 22, 2)]
wt_xpp_x = wt_xpp.iloc[:, range(0, 22, 2)]
fa_xpp_y = fa_xpp.iloc[:, range(1, 23, 2)] 
wt_xpp_y = wt_xpp.iloc[:, range(1, 23, 2)] 

fa_sim_act = fa_sim.iloc[:, range(8)] 
fa_sim_de = fa_sim.iloc[:, range(8, 11)]
wt_sim_act = wt_sim.iloc[:, range(8)]
wt_sim_de = wt_sim.iloc[:, range(8, 10)] 

wt_sim_act = wt_sim.iloc[:, range(8)]
wt_sim_de = wt_sim.iloc[:, range(8, 10)]
fa_sim_act= fa_sim.iloc[:, range(8)] 
fa_sim_de = fa_sim.iloc[:, range(8, 11)]  

#normalize xpp 
last_wt = wt_xpp.iloc[28000, 5] 
last_fa = fa_xpp.iloc[28000, 5] 

#plt.plot(wt_xpp.iloc[:, range(0, 18, 2)], fa_xpp.iloc[:, range(1, 19, 2)])

for i in range(8): 
    #wt_xpp_y.iloc[:, i] = wt_xpp_y.iloc[:, i].apply(lambda x : x / last_wt) 
    #wt_xpp_y.iloc[:, i].apply(lambda x : x / last_wt) 
    wt_xpp_y.iloc[:, i] = wt_xpp_y.iloc[:, i] / last_wt 
    #fa_xpp_y.iloc[:, i] = fa_xpp_y.iloc[:, i].apply(lambda x : x / last_fa) 
    #fa_xpp_y.iloc[:, i].apply(lambda x : x / last_fa) 
    fa_xpp_y.iloc[:, i] = fa_xpp_y.iloc[:, i] / last_fa 

print(fa_xpp_y.iloc[:, 10]) 
#normalize de 
last_wt = wt_xpp_y.iloc[0, 10] 
last_fa = fa_xpp_y.iloc[0, 10] 
for i in range(8, 11): 
    wt_xpp_y.iloc[:, i] = wt_xpp_y.iloc[:, i] / last_wt 
for i in range(8, 11):
    fa_xpp_y.iloc[:, i] = fa_xpp_y.iloc[:, i] / last_fa 

print(fa_xpp_y.iloc[:, 10]) 

wt_data_hds = wt_data_y.columns.tolist() #data 
fa_data_hds = fa_data_y.columns.tolist() 
fa_xpp_hds = fa_xpp_y.columns.tolist() #numerical 
wt_xpp_hds = wt_xpp_y.columns.tolist()  
wt_sim_hds = wt_sim.columns.tolist() #analytical 
fa_sim_hds = fa_sim.columns.tolist() 

f1 = plt.figure() 
ax1 = f1.add_subplot(111)

f2 = plt.figure() 
ax2 = f2.add_subplot(111) 

f3 = plt.figure() 
ax3 = f3.add_subplot(111) 

f4 = plt.figure() 
ax4 = f4.add_subplot(111) 

for i in range(8): 
    ax1.plot(wt_data_x.iloc[:, i], wt_data_y.iloc[:, i], c='y', ls='-', label=wt_data_hds[i]) #data 
    ax1.plot(wt_xpp_x.iloc[:, i], wt_xpp_y.iloc[:, i], c='b', ls='--', label=wt_xpp_hds[i], lw=3 ) #numerical 
    ax1.plot(wt_sim_act, c='r', label=wt_sim_hds[i]) #analytical 
    
    ax2.plot(fa_data_x.iloc[:, i], fa_data_y.iloc[:, i], c='y', ls='-', label=fa_data_hds[i]) 
    ax2.plot(fa_xpp_x.iloc[:, i], fa_xpp_y.iloc[:, i], c='b', ls='--', label=fa_xpp_hds[i], lw=3 ) 
    ax2.plot(fa_sim_act, c='r', label=fa_sim_hds[i])      
    
    #labels
    
for g in range(8, 10):
    ax3.plot(wt_data_x.iloc[:, g], wt_data_y.iloc[:, g], c='y', ls='-', label=wt_data_hds[g]) 
    ax3.plot(wt_xpp_x.iloc[range(2001), g+1], wt_xpp_y.iloc[range(2001), g+1], c='b', ls='--', label=wt_xpp_hds[g], lw=3  ) 
    ax3.plot(wt_sim_de, c='r', label=wt_sim_hds[g]) 
    
    #labels 
    
for h in range(8, 11): 
    ax4.plot(fa_data_x.iloc[:, h], fa_data_y.iloc[:, h], c='y', ls='-', label=fa_data_hds[h]) 
    ax4.plot(fa_xpp_x.iloc[:, h], fa_xpp_y.iloc[:, h], c='b', ls='--', label=fa_xpp_hds[h], lw=3 ) 
    ax4.plot(fa_sim_de, c='r', label=fa_sim_hds[h]) 
    
    #labels 
    
#print(wt_data_hds)
print(wt_xpp_hds)     
print(fa_xpp_hds)     

ax1.set_title('WT activation', fontsize=20, fontweight='bold')
ax2.set_title('FA activation', fontsize=20, fontweight='bold') 
ax3.set_title('WT deactivation', fontsize=20, fontweight='bold') 
ax4.set_title('FA deactivation', fontsize=20, fontweight='bold') 

for x in [ax1, ax2, ax3, ax4]: 
    x.set_ylabel('Normalized Open Probability', fontsize=14, fontweight='bold')
    x.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')

ax3.set_title('WT deactivation', fontsize=20, fontweight='bold') 

plt.show() 
