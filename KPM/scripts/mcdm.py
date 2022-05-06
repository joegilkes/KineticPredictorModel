import numpy as np

""" Functions for running Multi-Criteria-Decision-Making calculations
    This script ranks reactions using the minimum & maximum Ea as well 
    as the average & maximum dH.
"""

#####################################
## Functions for ranking reactions ##
#####################################

def ProcessArrays(EaMin,EaMax,EaAve,MaxdH,fstore,barrierless,sumdH):
    """
    Processes the prediction results and makes new arrays which are sorted based on the
    minimum activation energy.
    """
    nf = len(EaMin)
    Ea_min = np.zeros(nf)
    Ea_ave = np.zeros(nf)
    Ea_max = np.zeros(nf)
    dH_max = np.zeros(nf)
    sum_dH = np.zeros(nf)
    fil = []
    bar = []
    
    nrxn = len(EaMin)
    ic = 0
    for i in range(nrxn):
   # if EaMin[i] > 0:  this line means slightly negative barriers ignored - need to change....
        Ea_min[ic] = EaMin[i]
        Ea_max[ic] = EaMax[i]
        Ea_ave[ic] = EaAve[i]
        dH_max[ic] = MaxdH[i]
        sum_dH[ic] = sumdH[i]
        fil.append(fstore[i])
        bar.append(barrierless[i])
        ic += 1

    # The following does a bubble-sort based on the average barrier
    for i in range(nf):
        for j in range(i+1,nf):
            if Ea_min[j] < Ea_min[i]:
                t1 = Ea_min[j]
                t2 = Ea_min[i]
                t3 = Ea_ave[i]
                t4 = Ea_ave[j]
                t5 = Ea_max[i]
                t6 = Ea_max[j]
                t7 = dH_max[i]
                t8 = dH_max[j]
                t9 = sum_dH[i]
                t10 = sum_dH[j]
                f1 = fil[j]
                f2 = fil[i]
                b1 = bar[i]
                b2 = bar[j]
                Ea_min[j] = t2
                Ea_min[i] = t1
                Ea_ave[i] = t4
                Ea_ave[j] = t3
                fil[i] = f1
                fil[j] = f2
                bar[i] = b2
                bar[j] = b1
                Ea_max[i] = t6
                Ea_max[j] = t5
                dH_max[i] = t8
                dH_max[j] = t7
                sum_dH[i] = t10
                sum_dH[j] = t9
    
    return Ea_min, Ea_ave, Ea_max, dH_max, fil, bar, sum_dH
            

def MCDMarrays(Ea_min,Ea_ave,dH_max,Ea_max,sum_dH):
    """
    Construct multi-criteria decision making array.
    """
    nc = 0
    for i in range(nf):
        if bar[i]:
            nc += 1
    
    md = np.zeros([nc,nc])
    ic = 0
    for i in range(nf):
        if bar[i]:        
            jc = 0
            for j in range(nf):
                if bar[j]:
                    p1 = abs(Ea_min[i] / Ea_min[j])
                    #p2 = abs( Ea_ave[i] / Ea_ave[j])
                    p2 = abs( sum_dH[i]/sum_dH[j] )
                    p3 = abs( dH_max[i] / dH_max[j])
                    p4 = abs( Ea_max[i] / Ea_max[j])
                    md[ic,jc] = ( p1 * p2*p3* p4)
                    #md[ic,jc] = ( p1 * p3*   p4)
                    jc += 1
            print("* W values: ",ic,fil[i])
            print(md[ic,:],"\n")
            ic += 1
    return md, nc

def GetWinners(md,nc):
    """
    Based on the multi-criteria comparison matrix md[nc,nc], calculates how many mechanisms
    each mechanism "wins" against.
    
    md needs to have been calculated beforehand.
    
    nc is the number of mechanisms being assessed.
    """

    wins = np.zeros(nc,dtype=int)
    for i in range(nc):
        for j in range(nc):
            if i != j:
                if md[i,j] < 1.0:
                    wins[i] += 1
        print("* Winners: ",i,wins[i])
    return wins
