'''
1. open input data csv/xlsx files as dataframes 
2. separate activation and deactivation data 
3. allow user input to define how many data points to reduce (argv[0]) 
'''
import sys 
import glob
import math 
import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
from subprocess import call 

xl_path = r'C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/wai/excel/WT-act'

#convert xlsx files to csv 
#only works if the excel files are in the same directory as this python script 
def convert_to_csv(path):
    #path to conversion script 
    csv_path = xl_path + "/exceltocsv.bat" 
    call(csv_path) 
    #exit() #uncomment this if all you want is to convert 
    
#open each dataframe from csv or xlsx input and append to a list 
def get_df( path, type, reduce ):    
    #list to contain all dataframes 
    df_list = [] 
    #list to contain file names 
    file_names = [] 
    
    #read excel or csv separately 
    if type == 'xlsx': 
        data = glob.glob(path + '/*.xlsx')
        for d in data:        
            file_name = os.path.basename(d)[-22:-9] 
            file_names.append(file_name) 
            
            df = pd.read_excel(d, header=None) 
            #reduce number of data points by specified integer
            if reduce != 0: 
                dx = int(reduce) 
                df = df.iloc[ range(0, df.shape[0], dx), : ]
            df_list.append(df) 
        
    if type == 'csv':
        data = glob.glob(path + '/*.csv')
        for d in data:        
            file_name = os.path.basename(d)[-22:-9] 
            file_names.append(file_name) 
            
            df = pd.read_csv(d, header=None)
            df = df.iloc[ range(1, df.shape[0], 1), : ]
            #reduce number of data points by specified integer
            dx = int(reduce)
            if dx != 0: 
                print(dx) 
                df = df.iloc[ range(1, df.shape[0], dx), : ]
            df_list.append(df) 
        
    #quality check 
    if len(data) != len(df_list):
        print('number of dataframes != number of input spreadsheets') 
        exit() 
        
    return df_list, file_names  
    
#separate activation and deactivation in each dataframe 
def sepper( df, time, dx ): 
    #dx was used to reduce number of datapoints
    #determine new timestep 
    if dx != 0: 
        dt = 0.2*dx #new time interval 
    else:
        dt = 0.2 
    
    current = df.iloc[:, range(1, 10)] 
    volts = df.iloc[:, range(10, 19)] 
        
    #activation and deactivation containers 
    total_init = [] 
    total_act = []
    total_de = [] 
    
    #for p in range( len(current.columns.values.tolist()) ):
    for p in range(9):
        #current magnitude
        ci = [float(c) for c in current.iloc[:, p].values.tolist()]
        #test voltages ``   
        vt = [float(v) for v in volts.iloc[:, p].values.tolist()]
        
        t1 = [] 
        t2 = []
        t3 = [] 

        #given this protocol, the minimum dV = 35mV 
        #thus, find time points t1 and t2, 
        #where abs(t2 - t1) <= 0.2, and 
        #the voltages v(t1) and v(t2) satisfy:
        # abs( v(t1) - v(t2) ) >= 30mV 
        for u in range( 0, len(vt) - 10 ): #every 0.1s 
            
            vt1 = vt[u] 
            vt2 = vt[u + 10]               
       
            if abs( vt1 - vt2 ) >= 30: 

                #for initial hpol pulse 
                #if abs(vt1) < 0.5 and vt2 < 0 and len(t1) < 1: 
                if abs(vt1) < 0.5 and vt2 < 0:                    
                    t1.append(u) 
                    
                #end of hyperpolarization (s.t. t1 and t2 demarcate the activation trace) 
                #if vt1 < -10 and vt2 > -10 and len(t2) < 1: 
                if vt1 < -10 and vt2 > 0: 
                    t2.append(u) 
                    
                #end of initial depolarization (s.t. t2 and t3 demarcate the tail current)  
                #if vt1 > 10 and vt2 < 10 and len(t3) < 1: 
                if vt1 > 10 and vt2 < 10: 
                    t3.append(u) 
            
            else:
                continue 
        
        if len(t1) + len(t2) + len(t3) > 2: 
            #use median instead of mean to avoid indexing nonexistent timepoints 
            t_list = [int(np.median(x)) for x in [t1, t2, t3]] 

            #print( len(t1), len(t2), len(t3) )
            
            '''
            activation = t_list[0] : t_list[1] 
            deactivation = t_list[1] : t_list[2] 
            
            We can use these values to index time, current, and voltage equally because they are indices rather than absolute time values. 
            
            For each trace, there will be 3 columns: time, current, voltage. 
            
            100 x-units are removed on top of the sliced time to remove remaining capacitative transients. 
            '''
            print(t_list) 
            
            if dx != 0: 
                cap_cutoff = 100/dx 
            else:
                cap_cutoff = 100 
            
            total_act.append( time[t_list[0] : t_list[1]][cap_cutoff:] ) 
            total_de.append( time[t_list[1] : t_list[2]][cap_cutoff:] ) 
            
            total_act.append( ci[t_list[0] : t_list[1]][cap_cutoff:] )  
            total_de.append( ci[t_list[1] : t_list[2]][cap_cutoff:] ) 
            
            total_act.append( vt[t_list[0] : t_list[1]][cap_cutoff:] ) 
            total_de.append( vt[t_list[1] : t_list[2]][cap_cutoff:] ) 
        
    s_act = pd.DataFrame.from_records( total_act ).T 
    s_de = pd.DataFrame.from_records( total_de ).T 
    
    #return s_act, s_de 
    return s_act, s_de 
    
print(len(sys.argv))
if len(sys.argv) != 1:
    k = int( sys.argv[1] )
    print(k) 
else:
    k = 10 

df_l, df_labels = get_df( xl_path, 'csv', k )     
time = df_l[0].iloc[:, 0].values.tolist()

dfsep_l = [] 
for i in range( len(df_l) ):
    d1, d2 = sepper( df_l[i], time, k) 
    
    dfsep_l.append( d1 ) #activation 
    dfsep_l.append( d2 ) #deactivation 
    
n_dfsep = len(dfsep_l)

'''
To process tail currents:
1. subtract steady-state value ( take average of last 20 x-units for each deactivation trace, then the average for all of these; A1 ) 
2. collect the maximal tail current amplitude for each trace, then divide all by the max of this set (A2) 
''' 

bad = [] 
for t in range( 1, n_dfsep, 2 ): 
    a1 = [] 
    ml = [] 
    
    df = dfsep_l[t] 
    df_act = dfsep_l[t-1]
    for i in range(1, len(df.columns.values.tolist()), 3): 
        df_i = df.iloc[:, i] 
        a1.append( np.mean( df_i.values.tolist()[-20:] ) )
        ml.append( max( df_i.iloc[0:100].values.tolist() ) ) 
    
    if sum( ml[0:3] ) < sum( ml[3:] ):
        bad.append(t) 
    
    ml_n = [x/max(ml) for x in ml] 
    
    i_ss = min( df_act.iloc[:, 1].values.tolist() )
    for i in range(1, len(df.columns.values.tolist()), 3): 
        #normalize deactivation 
        df.iloc[:, i] = ( df.iloc[:, i] - a1[ int( (i-1) / 3 ) ] ) / max(ml) 
        f_inf = max( df.iloc[0:100, i].values.tolist() ) 
        
        #normalize activation traces 
        i_leak = np.mean( df_act.iloc[0:50, i].values.tolist() ) 
        f = ml_n[ int( (i-1) / 3 ) ] / i_ss 
        
        df_act.iloc[:, i] = (df_act.iloc[:, i]  - i_leak) * f
    
#remove traces where largest hyperpolarization does not produce largest tail current
print(len(bad))
j = 1 
for b in bad: 
    del dfsep_l[b-j] 
    del dfsep_l[b-j] 
    j += 2 
 
#plot normalized activation traces 
for x in range( 0, len(dfsep_l), 2 ):
    n = len( dfsep_l[x].columns.values.tolist() ) 
    d = dfsep_l[x] 
    
    f, axs = plt.subplots(2, 1)
    if n != 0:
        axs[0].plot( d.iloc[:, range(0, n, 3)], d.iloc[:, range(1, n, 3)] ) 
        axs[1].plot( d.iloc[:, range(0, n, 3)], d.iloc[:, range(2, n, 3)] ) 
    
    f.suptitle( df_labels[ int(x/2) ] ) 
    
#plt.show()

#average activation traces 
'''
1. concatenate all the activation traces into a single dataframe 
2. for time, compare .iat[-1] to find the largest and use that 
3. average each row of the activation traces 
'''
#put all the activation dataframes into a list 
n_act = [dfsep_l[i] for i in range(0, len(dfsep_l), 2)]

#number of columns in a single trace 
n = len( n_act[0].columns.values.tolist() ) 

#get time
time = df_l[0].iloc[:, 0]

#get row averages 
nd = {} 
for i in range(1, n, 3):
    #contain the activation traces for one given voltage 
    tr = [d.iloc[:, i] for d in n_act] 
    #combine into a single dataframe 
    df_merge = pd.concat( tr, axis = 1 )     
    #compute row average 
    row_avg = df_merge.mean(axis=1)
    #append row averages to a dictionary 
    j = int( (i-1)/3 ) 
    nd.update( { j : row_avg } ) 
    
#get individual dfs from the nd 
df_avg_l = [] 
df_avg_l.append(time)
for v in nd.values():
    df_avg_l.append( v ) 
    
#combine into a single dataframe 
df_avg_merge = pd.concat( df_avg_l, axis=1 ) 

#plot 
f = plt.figure() 
f.suptitle('Merged') 
ax = f.add_subplot(111) 
for x in range( 1, len(df_avg_merge.columns.values.tolist()) ):
    y = df_avg_merge.iloc[:, x].values.tolist()
    t_v = [i/5 for i in range(len(y))] 
    
    #clean up data to make sure there's no jumps at the end 
    #if y[-1] < y[-100]:
    del y[-100:]
    del t_v[-100:]
    
    #ax.plot( df_avg_merge.iloc[:, 0], df_avg_merge.iloc[:, x] ) 
    ax.plot( t_v, y ) 
plt.show() 
    


