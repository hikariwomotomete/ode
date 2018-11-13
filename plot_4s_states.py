#always begin your python code by importing any packages you use 
#for large libraries, it is more time-efficient to import specific packages from the libraries 
#this can be done via 'from X import Y' or directly specifying the specific package, e.g. 'import X.A,' where the '.' implies that A is a package contained in parent package X 
#to simplify 'calling' (using) packages while you code, you use 'as' to abbreviate the full name of the package to something else. 
#standard abbreviations are: pandas = pd, numpy = np, matplotlib.pyplot = plt, sympy = sp, etc. 
#if you use an abbreviation, you must NEVER re-use that abbreviation as a variable later on. For instance, you can't assign plt = 10 (a variable named 'plt' with value '10') and later expect to use the functions of the package 'matplotlib.pyplot' because the package is no longer assigned to 'plt.' 
import glob
import os 
from matplotlib import cm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#parameters for wild-type (WT) data  
#in Python, you can assign values to parameters by the '=' sign. 
#if you want to reassign a new value to a variable, you can simply do so later in th code, e.g. a=1 followed by a=2 will be understood as a=2. 
#you may assign variables for pretty much anything: things called 'data structures' (formal formats that contain data), numbers, words, etc. 
a0=0.0975 
sa=12.9 
b0=1870 
sb=95 

c0=48e-5 #a0' 
sc=10.5 #sa' 
d0=19e-3 #b0' 
sd=38 #sb' 

g1=47e-4 
h1=10e-5 
g2=0.000005 
h2=0.023

'''
#parameters for mutant (F431A; FA) data 
#the three apostrophes (') are like # symbols in that they *comment* out code, but #s comment out single lines, whereas the three apostrophes comment out whole blocks of code 
a0=1e-6
sa=47.6/3 
b0=2e-3 
sb=4347.8/5 

c0=2.5e-6 #a0' 
sc=200/9 #sa' 
d0=3e-5 #b0' 
sd=179/6 #sb' 

g1=5e-4 
h1=1.5e-5 
g2=5.7e-7 
h2=2.85e-4
'''
#create empty dictionaries called nd4s and nss 
#dictionaries are denoted by curly brackets {} 
#lists are denoted by square brackets [] 
#dictionaries and lists are distinct, as you'll see later. 
#each entry of a dictionary has the form {key : value} and these individual entries are referred to as key-value pairs. 
#keys are generally alphanumeric (composed of letters and/or numbers), while values may be nearly anything: lists, other dictionaries, numbers, etc. 
#lists on the other hand have the form [a, b, c, d], where a, b, c, d may be nearly any object: numbers, letters, words, dictionaries, arrays, etc. 
nd4s = {}
nss = {} 

#define a function called frange that gives a list of numbers with non-integer step sizes
#the normal 'range' function does not accept non-integer steps, so I have this just in case I ever want to try decimal steps (e.g. 0.5) 
def frange(start, stop, step):
    s = start #assign the input value 'start' to the variable 's' 
    
    #below, 'if' and 'while' are pretty self-explanatory, but there are subtleties. 
    #for 'if' statements, you give some condition in the statement, and if the condition is met, the indented code is executed
    #for 'while' statements, you also give some condition, but the conditions are slightly different - the code is only executed 'while' the condition is met. 
    #thus, 'while' statements are a bit more limited than 'if'
    #you can think of 'while' and 'if' statements as carrying their normal meanings in English for the most part 
    if step < 0: 
        while s > stop:
            yield s #basically, return the value of s that fulfills the above conditions. There's more to it than this, but I'll leave it there. 
            s += step # the '+=' is a very useful implementation that simply adds the value on the other side (here, 'step') and assigns the final sum to the starting variable, 's' 
            s = round(s, 15) #rounds the first value 's' to a value of maximum '15' decimal points 
    if step > 0:
        while s < stop:
            yield s
            s += step
            s = round(s, 15)

#repeat the following block of code for voltage values in the specified range
#the range here gives the following list of voltages: -50, -60, -70, -80, -90, -100, -110, -120
#note that -130 is not included. The 'range' function includes your starting value (-50) and the 2nd last value before the 'final' value
#equivalently, if you input range(X, Y, Z), you get all values X + nZ for n = 1, 2, 3, etc., until you reach X + nZ = Y - Z. So, Y - Z is your final value. 
#the 'for' term denotes the beginning of what is called a 'for loop' in Python speak. It allows you to repeat some given code under some given conditions 
#the 'in' term allows the 'for loop' to execute the given code for all values of the defined variable(s) under the defined conditions 
#any 'for loop' must end with a colon and the code to be repeated in the 'for loop' must always be indented once relative to the initial 'for loop' statement 
#finally, the general structure of a 'for loop' is
#for X in Y: (code to be executed repeatedly given different values of X) 
#where, X is the variable of interest that is defined by the conditions Y, which is usually a list 
for vol in range(-50, -130, -10):
    
    #evaluates the following expressions using voltages defined above  
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2 = d0*np.exp(vol/sd)

    #transition rate matrix 
    #np.array creates a matrix (aka array) in the following format: 
    #np.array( [ [1], [1], [1] ] ). This would create a matrix with 3 rows and 1 column. 
    #np.array( [ [1, 2], [1, 2], [1, 2] ] ) creates a matrix with 3 rows and 2 columns. 
    #note that any function you don't recognize can be Googled and you'll get all the gory details. 
    #note below that you can actually separate functions that are contained in brackets/parantheses, as long as you maintain proper indenting and close the brackets/parantheses when you're done. 
    
    #in the following matrix, the rows (from top to bottom) correspond to the ODEs of: C1, C2, O3, O4 
    #similarly, the columns (from left to right) correspond to: C1, C2, O3, O4 
    #thus, you'll see that each column sums to 0, which is a characteristic of what is known as a 'singular' matrix, which has interesting properties 
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    EigD = [] #creates an empty list. This is sort of redundant. 
    EigD = [w for w in W] #creates a list populated by each variable in W, which is the list of eigenvalues computed above (line 118). 
    #note how line 119 is redundant, because EigD will be repopulated with different eigenvalues each time the for loop is repeated (i.e. using different values of 'vol') 
    
    #the eigenvalues and eigenvectors are needed to define how the system evolves, but we also need initial conditions to solve our ODE system 
    #the initial conditions (ICs) are found by solving the 'analytical solution' for the steady-state of our ODE system 
    #that is, we set our ODEs equal to 0 and algebraically manipulate them until we get expressions for each ODE 
    
    #here, notice that we are re-calculating parameter values a1, b1, a2, b2 at a different voltage than what was used in the for loop (and find the eigenvalues/eigenvectors) 
    #this is because we are trying to find the ICs of our system at a different voltage 
    #however, we don't need a for loop to find ICs at multiple different voltages, because we are interested in how our system evolves from the same ICs 
    
    #technically, the voltage of ICs is called the 'holding voltage' because the system is 'held' at this voltage prior to further voltage stimuli
    #experimentally, the 'holding voltage' is held for long enough that the system reaches 'steady state,' which we calculate using the 'analytical solutions' as described above
    #once the 'steady state' of the system at a given 'holding voltage' is attained, a 'voltage pulse' (aka 'voltage step') is applied 
    #the 'voltage step' is the 'vol' we defined in our for loop initially, and are the voltages that will define how the system evolves out of the ICs 
    
    #here, our holding voltage is 0 (millivolts, mV), and we will be evaluating the response of the system at the voltages -50, -60, -70, ... -100, -110, -120. 
    
    v = 0
    a1=a0*np.exp(-v/sa) 
    b1=b0*np.exp(v/sb) 
    a2=c0*np.exp(-v/sc) 
    b2 = d0*np.exp(v/sd)    
    
    #LHS terms
    #this is an algebraic artifact that is hard to understand without the equations in front of you 
    #essentially, I am assigning the diagonal values of 'Q' (the matrix defined above) to certain variables ('alpha,' 'beta,' etc.) for convenience 
    #mathematically, this results from setting the ODEs to 0 and rearranging the equations so that both sides are positive. 
    alpha = a1 + g2 
    beta = b1 + g1
    gamma = b2 + h1 
    delta = a2 + h2 
    
    #what follows is a bunch of algebra that won't make much sense
    #don't worry about this stuff and just move on until you see a triple '###' 
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
        
    d4s = {} 
    d4s = { 'c1' : c1_4s, 'c2' : c2_4s, 'o3' : o3_4s, 'o4' : o4_4s}
    nd4s.update({ vol : d4s })
    
    ###
    ###
    ###
    #o4_4s, c1_4s, c2_4s, and o3_4s are the final 'steady state solutions' that were mentioned above 
    #essentially, these are expressions of the steady state populations of each state in our model 
    #notice that these expressions are an algebraic mix of parameters, nothing fancy 
    
    #below, we are making a list of the steady states after having computed them 
    #in the right context (as we'll see below), this list will be interpreted as a vector with four components
    #the order of the elements is therefore very important 
    SS_4s = [c1_4s, c2_4s, o3_4s, o4_4s]
    
    #we now use a linear algebra package (used above as well) in Numpy (np) to solve the canonical equation Ax = b 
    #where, A is a matrix, x and b are vectors 
    #in linear algebra, if you are multiplying two vectors or matrices, both components have to fulfill certain 'shape' conditions
    #simplistically, the term on the left (e.g. A) must have the same number of columns as the number of rows of the term on the right (e.g. x) 
    #that is a pretty difficult way to remember things. Instead, recall that when you see dimensions described as 'm x n,' m is the number of rows, and n the number of columns (e.g. m the height and n the width of a matrix) 
    #then, for a matrix A of size 'm x n,' and a vector x of size 'a x b,' the operation (multiplication) of A*x is only possible if n = a. 
    #the dimensions of this multiplication, as a rule, will be 'm x b.' 
    
    #that's quite a bit of linear algebra in a nutshell
    #as mentioned, we are solving the equation Ax = b, knowing A and b, for an unknown 'x' 
    #where, V (our eigenvectors computed from Q, line 118) is A and SS_4s (our steady state solutions, line 193) is b 
    
    #thus, 'x' is a vector that, when multipled by the eigenvectors in V, produce our steady states in SS_4s 
    #we are solving for constants in the intial value problem (IVP) defined by our ODEs and the steady state solutions, which are assumed to be the solutions of our ODEs at time 0 
    #see this page for more info: http://tutorial.math.lamar.edu/Classes/DE/RealEigenvalues.aspx
    const = np.linalg.solve(V, SS_4s)
    
    
    #analytical simulations
    #make a new dictionary called 'ss_l' that we will use to store our simulation data 
    ss_l = {}
    
    #simulate our ODEs for time 'x' from x=0 to x=7300 (no units specified here, but they are in milliseconds; ms) 
    for x in range(0, 7310, 10):
    
        #we can implement a for loop in a single expression (usually a list or within another function - here it is 'sum') 
        #using a for loop, we are computing the solutions for each of the ODEs and then summing all four terms 
        #notice the only difference between ODE solutions is the eigenvector V[n, i], where i is iterated through 0, 1, 2, 3 due to how we ordered our matrices, states, etc. previously 
        #n is also determined by how we ordered things; 0 is c1, 1 is c2, 2 is o3, and 3 is o4 
        
        cr4s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 4))
        ca4s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 4))
        or4s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 4))
        oa4s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 4))

        #totals
        C_t = cr4s + ca4s 
        O_t = or4s + oa4s 
        states = [cr4s, ca4s, or4s, oa4s]
        ss_l.update( {x : states} ) #populate the dictionary we made with a separate key:value entry for each unique time value
   
    nss.update( {vol : ss_l} )  #populate the dictionary nss with a key:value entry of the step voltage 'vol' as the key and the dictionary containing {time:state population} entries as the values. this is known as a 'nested dictionary' because it is a dictionary whose values comrpise another dictionary 
    
    
#repetition of the for loop above for different holding and step voltages 
for vol in [0, 20]:

    #unbound 
    a1=a0*np.exp(-vol/sa) 
    b1=b0*np.exp(vol/sb) 
    a2=c0*np.exp(-vol/sc) 
    b2=d0*np.exp(vol/sd)
    
    Q = np.array([
                 [-(a1 + g2), b1, h2, 0],
                 [a1, -(b1 + g1), 0, h1],
                 [g2, 0, -(a2+h2), b2],
                 [0, g1, a2, -(b2 + h1)]
                 ]
                 )
    #print(Q.shape)
    W, V = np.linalg.eig(Q) #returns eigenvalues in W, right eigenvectors in V such that V[:,i] corresponds to W[i]
    #EigD = [W[1,1], W[2,2], W[3,3], W[4,4], W[5,5], W[6,6], W[7,7], W[8,8]]
    EigD = []
    EigD = [w for w in W]
    #print(EigD)
    #print(W.shape)
    
    #activation - held at -40, deactivation held at -140 
    #unbound 
    v = -100
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
        
    d4s = {} 
    d4s = { 'c1' : c1_4s, 'c2' : c2_4s, 'o3' : o3_4s, 'o4' : o4_4s}
    nd4s.update({ vol : d4s })
    
    SS_4s = [c1_4s, c2_4s, o3_4s, o4_4s]
    
    const = np.linalg.solve(V, SS_4s)
    
    #analytical simulations
    ss_l = {}
    for x in range(0, 510, 10):
        cr4s = sum(const[i] * np.exp(EigD[i]*x) * V[0,i] for i in range(0, 4))
        ca4s = sum(const[i] * np.exp(EigD[i]*x) * V[1,i] for i in range(0, 4))
        or4s = sum(const[i] * np.exp(EigD[i]*x) * V[2,i] for i in range(0, 4))
        oa4s = sum(const[i] * np.exp(EigD[i]*x) * V[3,i] for i in range(0, 4))

        #totals
        C_t = cr4s + ca4s 
        O_t = or4s + oa4s 
        states = [cr4s, ca4s, or4s, oa4s]
        ss_l.update( {x : states} ) 
   
    nss.update( {vol : ss_l} )

dss = pd.DataFrame.from_dict(nss) #create a dataframe from the nested dictionary nss 
# a dataframe is a unique data structure 
# dataframes are mostly used with the package 'Pandas' (pd) 

heads = dss.columns #not really useful, but this is a list of the column labels of 'dss' 
n = len(dss.columns.values) #this is the number of items in the list above 

for x in range(0, 10, 2):
    
    f = plt.figure() #make an empty figure for each iteration of x 
    ax1 = f.add_subplot(211) #create a subplot within a 2x1 grid at the 1st position within the grid 
    ax2 = f.add_subplot(212) #create a subplot at the 2nd position of the 2x1 grid 
    
    full_1 = dss.iloc[:, x].apply(pd.Series) 
    #iloc[:, n] extracts all rows (:) of a given column (x), where x is a number that cannot exceed or be below the number of columns in the dataframe (dss)
    #apply(pd.Series) just turns the extractant (the result of the iloc operation) into another dataframe known as a Series 
    
    #equivalently, I could have split line 338 into two lines: 
    #a = dss.iloc[:, x], then 
    #b = a.apply(pd.Series) 
    
    closed_1 = full_1.iloc[:, 0:2] #the same iloc operation as above, except the columns extracted are given by a 'slice' 
    #slices are denoted by two numbers separated by a colon (:) 
    #slices are like the range function
    #slices take the first number and every number up to, but not including, the last 
    #there is no 'step size,' unlike the range function. The default (and only) 'step' is one, so a slice of 23:26 would give [23, 24, 25] 
    #if there is a need to skip columns or otherwise specify columns in a way that isn't satisfied by the slice operation, the columns can be explicitly defined
    #this is done by putting the columns in a list, e.g. [1, 5, 7, 9] 
    #in the iloc operator: full_1.iloc[:, [1, 5, 7, 9]] 
    #notice that square brackets are required to enclose the list above but not required for slices 
    
    open_1 = full_1.iloc[:, 2:4]
    
    l_1c = ax1.plot(closed_1, ls='--') #plot(a, b) is a function of matplotlib.pyplot and plots 'a' info on the x-axis and 'b'-info on the y-axis. 
    #ls is short for linestyle, and '--' indicates that we want the lines to be shown as dashes 
    l_1o = ax1.plot(open_1) 
    #notice that we don't actually haven't passed explicit (x, y) information to the plot function
    #this is because the dataframes open_1 and closed_1 are data structures that can be conveniently interpreted by the package automatically for x,y info if appropriately constructed 
    
    #dataframes 
    #dataframes are essentially large matrices with interesting and convenient data handling properties
    #the very top row of a dataframe is an optional row for column labels, called 'headers' 
    #the very first column of a dataframe is known as an 'index,' and essentially numbers each row. However, the index may be optionally deleted or modified to include non-standard values (e.g. words), or use custom numerics (e.g. step sizes that are not one). By default, the index simply numbers rows from 1, 2,...n, where n is the number of rows. 
    #if a particular cell is empty, the dataframe fills it with 'NaN,' which is interpreted as zero in most applications 
    #however, 'NaN' can be explicitly set to '0' instead. See documentation online for more information. 
    
    #when we plot dataframes, the index is automatically interpreted as the x-values and the values in each column are considered distinct sets of y-values. Thus, each column represents a separate curve, if we are plotting more than one column of a dataframe. 

    j = x + 1 
    full_2 = dss.iloc[:, j].apply(pd.Series)
    closed_2 = full_2.iloc[:, [0,1]]
    open_2 = full_2.iloc[:, [2,3]]
    l_2c = ax2.plot(closed_2, ls='--')
    l_2o = ax2.plot(open_2)
    
    #just some fancy graphics stuff 
    #in this case, setting up our legend 
    leg1 = ax1.legend(l_1c, ['cr', 'ca'], loc=1, borderaxespad=0., bbox_to_anchor=(1.08, 1), fontsize='large')
    leg2 = ax1.legend(l_1o, ['or', 'oa'], loc=1, borderaxespad=0., bbox_to_anchor=(1.08, 0.7), fontsize='large')
    leg_2 = ax1.add_artist(leg2)
    leg_1 = ax1.add_artist(leg1)
    
    #self-explanatory labelling stuff 
    plt.xlabel('Time in ms', fontsize=12)
    ax1.set_ylabel('State Population', fontsize=12)
    ax2.set_ylabel('State Population', fontsize=12)
    ax1.set_title(str(dss.columns.values[x]))
    ax2.set_title(str(dss.columns.values[j]))

    plt.show() #this is required to actually have any figures made by matplotlib to show up
    
