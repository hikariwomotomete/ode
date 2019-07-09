#1. read activation excel files from Wai data 
#2. select time window of interest (optional) 
#3. compute first and second derivatives 
#4. clean up the data using moving averages 

import glob
import math 
import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import *
from scipy.interpolate import * 

xl_path = r'C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\data_files\wai\excel'

def get_df( path ):
    data = glob.glob(path + '/110316_0028.xlsx') 
    
    n = len(data) #number of excel files 
    if n < 2: 
        for d in data:         
            df = pd.read_excel(d) 
        return df 
    else:
        df_list = [] 
        for d in data:
            df = pd.read_excel(d) 
            df_list.append(df) 
        dfmerge = pd.concat(df_list, axis=1) 
        return dfmerge 
         
#select activation traces; exclude tail currents for now 
def act_only( data_df, trace_num, t_start, t_end ):
    #trace_num is the number of traces 
    #data_df contains the output of get_df 
    df = data_df 
    n = trace_num 
    
    #t_start and t_end define time window 
    t1 = int( t_start*5 ) 
    t2 = int( t_end*5 ) 
    
    #get traces in the desired time window 
    #column selection requires n+1 because the first column in the dataframe of data is time values 
    df_t1t2 = df.iloc[t1:t2+1, 1:n+1]
        
    #filter out tail currents 
    data_xval = range(t1, t2+1) 
    new_records = [] 
    
    #p = [] 
    for i in range(n):
        #list of current magnitude for each trace 
        data_yval = df_t1t2.iloc[:, i].values.tolist()
        
        for x in range( len(data_yval) ):    
            # if you only have one column in a dataframe, iloc only needs the row index 
            #iloc returns a series if only using one value, so use iat instead 
            
            #index of each y value 
            #y = data_yval.iat[x]
            
            #current magnitude
            y = data_yval[x]
            if not math.isnan(y): 
                if y > 0:
                    #x is row number, i is column number
                    df_t1t2.iat[ x, i ] = math.nan 
                    #print(y, df_t1t2.iat[ x, i ] ) 
                if y <= 0: 
                    pass     
    
    #delete rows that correspond to elements of list p
    #better to make all positive (tail current) values nan
    #df_t1t2.drop( df_t1t2.index[p] , axis=0 , inplace=True ) 
    
    return df_t1t2
    
#compute moving average for a given dataframe 
def smooth(x, filt_degree):
    #filt_degree is the number used to determine 'averaging width' 

    #x is a dataframe containing corresponding time and current values 
    
    hd_num = len( x.columns.values.tolist() ) 
    idx = x.index.tolist() 
    
    fin = [] #to contain individual x, y lists 
    for h in range(hd_num): 
        yv = [i for i in x.iloc[:, h].values.tolist() if not math.isnan(i)] 
        xv = [idx[i] for i in range(len(yv)) if not math.isnan(idx[i]) ] 
        n = len(yv)
        
        p = [] 
        for i in range(0, n, filt_degree): 
            if i < filt_degree: 
                sub = yv[i : i + filt_degree + 1]
            elif filt_degree < i < n - filt_degree: 
                sub = yv[i-filt_degree : i + filt_degree + 1]
            elif i > n-10:
                sub = yv[i - filt_degree : n] 
                
            p.append( np.mean(sub) ) 
        
        new_x = [xv[j] for j in range(0, n, filt_degree)] 
    
        fin.append(new_x) #new x values 
        fin.append(p) #averaged y values 
        
    return pd.DataFrame.from_records(fin).T 
  
#finite differences using central formula 
def finiter(df):
    #find deltax by subtracting two index values 
    idx = df.index.values.tolist()
    deltax = int( idx[1] - idx[0] ) 
    
    #for each column, collect finite differences in c 
    c = [] 
    for u in range(0, len(df.columns.values.tolist()), 2): 
           
        #open respective time and current values    
        xs = df.iloc[:, u].values.tolist() 
        ys = df.iloc[:, u+1].values.tolist() 
        
        h = range(len(ys))
        ds = [] 
        
        #for each current magnitude, compute finite diff 
        for g in h:
            if g > 0 and g < h[-1]: 
                d = (1/ (2*deltax)) * (ys[g+1] - ys[g-1])
            elif g == h[-1]: 
                d = (1/deltax) * (ys[g] - ys[g-1]) 
            elif g == 0:
                d = (1/deltax) * (ys[g+1]-ys[g]) 
                
            if d <= 0: 
                #append finite diff to a list 'ds' 
                ds.append(d) 

        c.append(xs) 
        c.append(ds) 
        
    return pd.DataFrame.from_records(c).T 
       
def plotter(df, leg, name, line):
    #line is list
    #line[0] = list of time values for max 1st dv 
    #line[1] = list of curves for which this line should be plotted 

    n = len(df.columns.values.tolist())
    
    if leg != None: 
        labs = leg 
    else:
        labs = list(range(n))
    
    f, ax = plt.subplots(1, 1)
    f2, ax2 = plt.subplots(1, 1)
    for i in range(0, n, 2):            
        ax.plot( df.iloc[:, i], df.iloc[:, i+1], label=labs[i] ) 
        ax2.plot( df.iloc[:,i], savgol_filter( df.iloc[:, i+1], 51, 3 ), label=labs[i] ) 
        
    if line != None:
        for t in line[1]: 
            #get y values 
            y = df.iloc[:, (2*t)+1].values.tolist()
            
            #plot lines
            #to plot the tth line defined by 't in line[1]'
            ax.plot( [line[0][t]]*100, np.linspace( np.nanmin(y), np.nanmax(y), 100 ), ls='--', lw=2)   
            ax2.plot( [line[0][t]]*100, np.linspace( np.nanmin(y), np.nanmax(y), 100 ), ls='--', lw=2)
            
    plt.legend() 
    
    if name != None: 
        f.suptitle('%s (no filter)' % (name,) ) 
        f2.suptitle('%s (filtered)' % (name,) )
    else:
        f.suptitle('nofilter')
        f2.suptitle('filtered') 
    plt.show() 
    
def ma(df):
    #assume even are time, odd are current 
    n = len( df.columns.values.tolist() ) 
    
    xy = [] 
    for t in range(0, n, 2): 
        xy.append( df.iloc[:, t].values.tolist() ) 
        xy.append( df.iloc[:, t+1 ].rolling(5).mean().values.tolist() )
        
    df_ma = pd.DataFrame.from_records(xy).T
    
    return df_ma 

def maxer( df ):
    n = len( df.columns.values.tolist() ) 
    
    g = [] #collection of maximal values 
    for f in range(0, n, 2):
        y = df.iloc[:, f+1].values.tolist() 
        m = np.nanmin(y) 
        for i in range(len(y)):
            if y[i] == m: 
                g.append(i) 
    
    xs = [] 
    for i in range(len(g)):
        xs.append( df.iloc[:, i*2].values.tolist()[g[i]] ) 
    return xs 
    
data = get_df(xl_path) 
filt_data = act_only( data, 9, 816.6,  3824.4) 
print(filt_data.iloc[:, [-2, -1]])

df_avg = smooth( filt_data, 30 ) 

df_fin = finiter(df_avg) 

#compute and plot first derivative 
df_fin_ma = ma(df_fin) 
print(df_fin_ma) 
#find max value for 1st dv 
mx_1dv = maxer(df_fin_ma) 
print( [x/5 for x in mx_1dv] )
#plot 
mx_line = [mx_1dv, list(range(len(mx_1dv))) ] 
plotter(df_fin_ma, None, None, mx_line)

#compute and plot the second derivative 
df_2fin = finiter(df_fin_ma) 
df_2fin_ma = ma(df_2fin) 
#plotter(df_2fin, None, 'df_2fin') 
#plotter(df_2fin_ma, None, 'df_2fin_ma') 
