# mosum
# anwendbar: 
##  N = mosum(x, ...), changepoints sind in N['cpts']
##  plot_mosum(x, ...) oder plot_mosum(N, ...)
##  F = multiscale_bottomUp(x, G = [Liste von Bandbreiten], ...)
##  Plot hiervon auch via plot_mosum(F, ...) (einer der Plots ist leer, aber der unrelevantere)

## multiscale_localPrune existiert, tut aber leider nicht das, was es soll, vermutlich stimmen irgendwo Indexverschiebungen nicht
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
def rollingSum(x, G):
    n = len(x)
    res = np.zeros(n) + np.nan
    currentSum = 0
    for j in range(G):
        currentSum += x[j]
    res[0] = currentSum
    for j in range(1, n-G+1):
        currentSum += x[j+G-1]
        currentSum -= x[j-1]
        res[j] = currentSum
    return(res)
#hier scheint man mir noch ein paar Zeilen einsparen zu können

def eta_criterion_help(candidates, mvalues, eta, GLeft, GRight):
    n = len(mvalues)
    res = []
    leftLength = int(eta*GLeft)
    rightLength = int(eta*GRight)
    for j in range(len(candidates)):
        kstar = candidates[j]
        mstar = mvalues[kstar]
        leftthresh = max(1, kstar-leftLength)
        rightthresh = min(n-1, kstar+rightLength)
        largest = True

        for l in range(leftthresh, rightthresh+1):
            if not largest: break
            if mvalues[l] > mstar: largest = False
            
        if largest: # so kstar would be the maximum in its eta*G-environment and accepted as a changepoint
            res.append(kstar)
    return(res)
def mosum_stat(x, G, GRight = None, varEstMeth = 'mosum', varCustom = None, boundaryExtension = True):
    x = np.array(x)
    n = len(x)
    symmetric = pd.isna(GRight)
    assert not(G < 1 and G > 0.5), 'rel. bandwidth not between 0 and 0.5'
    assert not(G >= n/2), 'bandwidth not smaller than length(x)/2'
    if not symmetric:
        assert not(GRight < 1 and GRight > 0.5), 'rel. bandwidth not between 0 and 0.5'
        assert not(GRight >= n/2), 'bandwidth not smaller than length(x)/2'
        
    
    absBandwidth = (G >=1)
    if not absBandwidth:
        G = int(n*G)
        if not symmetric: GRight = int(n*GRight)
    
    # consistency checks on input
    assert len(x.shape) == 1, 'input vector not one-dimensional'
    assert G > 0 and G < n, 'bandwidth error'
    #assert symmetric or not pd.isna(GRight)
    assert symmetric or (GRight > 0 and GRight < n), 'bandwidth error'
    
    GLeft = G
    if symmetric: GRight = G
    Gmin = min(GRight, GLeft)
    Gmax = max(GRight, GLeft)
    K = Gmin/Gmax
    
    # calculate value of statistics
    sumsLeft = rollingSum(x, GLeft)
    if(GLeft == GRight):
        sumsRight = sumsLeft
    else:
        sumsRight = rollingSum(x, GRight)
        
    unscaledStatistic = np.concatenate([np.zeros(GLeft-1) + np.nan, Gmin/GRight*sumsRight[(GLeft):n] - Gmin/GLeft*sumsLeft[0:(n-GLeft)], [np.nan]]) / np.sqrt((K+1)*Gmin)
    
    # calculate variance estimation
    assert not(not pd.isna(varCustom) and varEstMeth != 'custom'), "use varEstMeth = 'custom' when parsing varCustom"
    if varEstMeth == 'custom':
        assert not pd.isna(varCustom), "varCustom must not be Null for varEstMeth = 'custom'"
        assert not len(varCustom) != n, "varCustom is not of length n = length x"
        var = varCustom
    elif varEstMeth == 'global': # apparently deprecated
        var = np.array([(sum(x**2)-(sum(x)**2)/n)/n] * n)
    else: # now the MOSUM variance estimators
        summedSquaresLeft = rollingSum(x**2, GLeft)
        squaredSumsLeft = sumsLeft**2
        varTmpLeft = summedSquaresLeft[0:(n-GLeft +1)] - 1/GLeft*squaredSumsLeft[0:(n-GLeft +1)]
        varLeft = np.concatenate([np.zeros(GLeft -1) + np.nan, varTmpLeft]) / GLeft
        if GLeft == GRight:
            summedSquaresRight = summedSquaresLeft
            squaredSumsRight = squaredSumsLeft
            varTmpRight = varTmpLeft
        else:
            summedSquaresRight = rollingSum(x**2, GRight)
            squaredSumsRight = sumsRight**2
            varTmpRight = summedSquaresRight[0:(n-GRight +1)] - 1/GRight*squaredSumsRight[0:(n-GRight +1)]
        varRight = np.concatenate([varTmpRight[1:(n-GRight+1)], np.zeros(GRight) + np.nan]) / GRight

        if varEstMeth == 'mosum': var = (varLeft + varRight)/2
        elif varEstMeth == 'mosum_left': var = varLeft
        elif varEstMeth == 'mosum_right': var = varRight
        elif varEstMeth == 'mosum_min': var = np.minimum(varLeft, varRight)
        elif varEstMeth == 'mosum_max': var = np.maximum(varRight, varLeft)
        else:
            print("unknown varEstMeth, default to 'mosum'")
            var = (varLeft + varRight)/2
    
    varEstimation = var
    
    # CUSUM extension to boundary
    if boundaryExtension:
        if n > 2*GLeft:
            weightsLeft = np.sqrt((GLeft + GRight) / np.arange(1, GLeft +1) /  np.arange(GLeft + GRight -1, GRight - 1, -1))
            unscaledStatistic[:GLeft] = np.cumsum(np.mean(x[0:GLeft+GRight]) - x[0:GLeft]) * weightsLeft
            varEstimation[:GLeft] = varEstimation[GLeft-1]
            
        if n > 2*GRight:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide") # last entry will get special treatment, ignore the warning
                weightsRight = np.sqrt((GLeft + GRight) / np.arange(GRight-1, -1, -1) / np.arange(GLeft+1, GLeft + GRight + 1))
            xrev = x[(n - GLeft - GRight):n]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message = "invalid value encountered in multiply")
                unscaledStatistic[(n-GRight):n] = np.cumsum(np.mean(xrev) - xrev)[-len(xrev)+GLeft:] * weightsRight
            unscaledStatistic[n-1] = 0
            varEstimation[(n - GRight):n] = varEstimation[n-GRight-1]
            
    res = abs(unscaledStatistic) / np.sqrt(varEstimation)
    
    retval = {'x':x, 'GLeft':GLeft, 'GRight':GRight, 'varEstMeth':varEstMeth, 'varCustom':varCustom, 'boundaryExtension':boundaryExtension,
              'stat': res, 'rollsums':unscaledStatistic, 'varEstimation':varEstimation}
    
    return(retval)
def asymptoticA(x):
    return(np.sqrt(2*np.log(x)))

def asymptoticB(x, K):
    return(2*np.log(x) + 0.5*np.log(np.log(x)) + np.log((K**2 + K + 1)/(K + 1)) - 0.5*np.log(np.pi))

def mosum_criticalValue(n, GLeft, GRight, alpha):
    Gmin = min(GLeft, GRight)
    Gmax = max(GLeft, GRight)
    K = Gmin/Gmax
    return((asymptoticB(n/Gmin, K) - np.log(np.log(1/np.sqrt(1-alpha)))) / asymptoticA(n/Gmin))

def mosum_pValue(z, n, GLeft, GRight = None):
    GRight = GRight or GLeft
    Gmin = min(GLeft, GRight)
    Gmax = max(GLeft, GRight)
    K = Gmin/Gmax
    return(1 - np.exp(-2*np.exp(asymptoticB(n/Gmin, K) - asymptoticA(n/Gmin)*z)))
def mosum(x, G, GRight = None, varEstMeth = 'mosum', varCustom = None, boundaryExtension = True, threshold = 'critical_value', alpha = 0.1, thresholdCustom = None, 
          criterion = 'eta', eta = 0.4, epsilon = 0.2, doConfint = False, level = 0.05, Nreps = 1000):
        
    GRight = GRight or G
    
    # consistency checks on input
    assert alpha >0 and alpha < 1
    assert criterion in ['epsilon', 'eta']
    assert criterion != 'epsilon' or epsilon > 0
    assert criterion != 'eta' or eta > 0
    assert not doConfint or Nreps > 0

    x = np.array(x)
    n = len(x)
    
    m = mosum_stat(x, G, GRight, varEstMeth, varCustom, boundaryExtension)
    GLeft = m['GLeft']
    GRight = m['GRight']
    Gmin = min(GLeft, GRight)
    Gmax = max(GLeft, GRight)
    K = Gmin/Gmax
    changepoints = []
    
    if threshold == 'critical_value' and Gmax/Gmin > 4:
        print('Warning: Bandwidths are too unbalanced, \n (G, GRight) satisfying max(G, GRight)/min(G, GRight) <= 4 is recommended')
    
    if threshold == 'critical_value':
        thresholdVal = mosum_criticalValue(n, GLeft, GRight, alpha)
    elif threshold == 'custom':
        thresholdVal = thresholdCustom
    else:
        print("threshold must be 'critical_value' or 'custom', default to 'critical_value'")
        thresholdVal = mosum_criticalValue(n, GLeft, GRight, alpha)
        
    exceedings = m['stat'] > thresholdVal

    if criterion == 'epsilon':
        exceedingsCount = [1]
        for i in range(1, len(exceedings)):
            if exceedings[i] != exceedings[i-1]:
                exceedingsCount.append(1)
            else: exceedingsCount.append(exceedingsCount[-1] + 1)
        exceedingsCount = exceedings * exceedingsCount
        
        minIntervalSize = max(1, (Gmin + Gmax)/ 2*epsilon)
        intervalEndpoints = [i for i, value in enumerate(np.diff(exceedingsCount) <= -minIntervalSize) if value == True]
        #print(intervalEndpoints)
        intervalBeginpoints = intervalEndpoints - exceedingsCount[intervalEndpoints] + 1
        #print(intervalBeginpoints)
        
        # Hier könnten sich potenziell schnell Indexfehler um 1 eingeschlichen haben. Die simple Lösung: Immer boundaryExtension = True wählen...
        if not m['boundaryExtension']:
            # adjust right border
            if exceedings[n-GRight-1] and not ((n-GRight-1) in intervalEndpoints):
                lastBeginpoint = n - GRight - exceedingsCount[n-GRight - 1]
                assert exceedings[np.arange(lastBeginpoint, n - GRight + 1)]
                assert not lastBeginPoint in intervalBeginPoints
                temp = m['stat'][np.arange(lastBeginpoint, n - GRight + 1)]
                highestStatPoint = [i for i, value in enumerate(temp)  if value == max(temp)][0] + lastBeginPoint - 1
                if highestStatPoint-lastBeginpoint >= minIntervalSize/2:
                    np.append(intervalEndpoints, n - GRight)
                    np.append(intervalBeginpoints, lastBeginPoint)
                    
            #adjust left border
            if exceedings[GLeft-1] and not GLeft in intervalBeginpoints:
                firstEndpoint = [i for i, value in enumerate(np.diff(exceedingsCount)) if value < 0][1]
                assert exceedings[np.arange(GLeft, firstEndpoint)]
                assert not firstEndpoint in intervalEndpoints
                temp = m['stat'][np.arange(GLeft, firstEndpoint + 1)]
                highestStatPoint = [i for i, value in enumerate(temp) if value == max(temp)][0] + GLeft - 1
                if firstEndpoint - highestStatPoint >= minIntervalSize/2:
                    np.append(intervalEndPoints, firstEndpoint)
                    np.append(intervalBeginpoints, GLeft)
        
        assert len(intervalBeginpoints) == len(intervalEndpoints), "something went terribly wrong"
        numChangepoints = len(intervalBeginpoints)
        if numChangepoints > 0:
            for i in range(numChangepoints):
                temp = m['stat'][np.arange(intervalBeginpoints[i], intervalEndpoints[i] + 1)]
                changepoint = intervalBeginpoints[i] + [i for i, value in enumerate(temp) if value == max(temp)][0]
                changepoints.append(changepoint)
                
                
    else: # criterion = 'eta'
        localMaxima = np.append(np.diff(m['stat']) < 0, False) & np.concatenate([[False], np.diff(m['stat']) > 0]) # testing if greater than both neighbours
        # adjust in case of no boundary cusum extension
        if not m['boundaryExtension']:
            localMaxima[n-GRight-1] = True
        
        pCandidates = [i for i, value in enumerate(exceedings & localMaxima) if value == True]
        changepoints = eta_criterion_help(pCandidates, m['stat'], eta, GLeft, GRight)
        
        
    retval = {'x':x, 'GLeft':GLeft, 'GRight':GRight, 'varEstMeth': m['varEstMeth'], 'varCustom': m['varCustom'], 'boundaryExtension': m['boundaryExtension'],
              'stat': m['stat'], 'rollsums': m['rollsums'], 'varEstimation': m['varEstimation'], 'threshold': threshold, 'alpha': alpha, 'thresholdCustom':thresholdCustom,
              'thresholdValue': thresholdVal, 'criterion': criterion, 'eta': eta, 'epsilon': epsilon, 'cpts': changepoints}    
    return(retval)
    
def plot_mosum(x, G = 0.2, GRight = None, varEstMeth = 'mosum', varCustom = None, boundaryExtension = True, threshold = 'critical_value', alpha = 0.1, thresholdCustom = None, 
          criterion = 'eta', eta = 0.4, epsilon = 0.2, doConfint = False, level = 0.05, Nreps = 1000):
    
    # x can be given as the result of an already completed mosum procedure or as the series and mosum will be executed here
    if type(x) != dict:
        N = mosum(x, G, GRight, varEstMeth, varCustom, boundaryExtension, threshold, alpha, thresholdCustom, criterion, eta, epsilon, doConfint, level, Nreps)
    else: N = x
    
    fig, ax = plt.subplots(2, figsize=[18, 8])
    data = N['x']
    ax[0].plot(data)
    chps = N['cpts']
    currchp = 0
    if len(chps) == 0:
        print('\n \t No changepoints were found.')
    else:
        for chp in chps:
            interv = data[currchp:chp]
            ax[0].axvline(chp, 0.05, 0.95, color = 'gold', linestyle = '--')
            ax[0].plot(range(currchp, chp+1), [np.mean(interv)]*(len(interv)+1), c = 'gainsboro', ls = ':')
            currchp = chp
    interv = data[currchp:]
    ax[0].plot(range(currchp, len(data)+1), [np.mean(interv)]*(len(interv)+1), c = 'gainsboro', ls = ':')
            
    ax[0].set_title('Time series data with changepoints')
    
    try:
        ax[1].plot(N['stat'], c = 'slategrey')    
        ax[1].axhline(N['thresholdValue'], color = 'cadetblue', linestyle = ':')
        for chp in chps:
            ax[1].axvline(chp, 0.05, 0.95, color = 'gold', linestyle = '--')
            
        ax[1].axhspan(N['thresholdValue'], max(N['thresholdValue']+0.3, max(N['stat'])+0.2), facecolor='ghostwhite', alpha=0.3)
        ax[1].set_title('Monitored statistics')
    except KeyError:
        pass
    
    plt.show()
Nile = [1120, 1160,  963, 1210, 1160, 1160,  813, 1230, 1370, 1140,  995,  935, 1110,  994, 1020,  960, 1180,  799,  958, 1140, 1100, 1210, 1150, 1250,
1260, 1220, 1030, 1100,  774,  840,  874,  694,  940,  833,  701,  916,  692, 1020, 1050,  969,  831,  726,  456,  824,  702, 1120, 1100,  832,
764,  821,  768,  845,  864,  862,  698,  845,  744,  796, 1040,  759,  781,  865,  845,  944,  984,  897,  822, 1010,  771,  676,  649,  846,
812,  742,  801, 1040,  860,  874, 848,  890,  744,  749,  838, 1050,  918,  986,  797,  923,  975,  815, 1020,  906,  901, 1170,  912,  746,
919,  718,  714,  740]
N = mosum(Nile, G = 25)
