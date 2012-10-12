import numpy as np
from scipy.stats import scoreatpercentile
from matplotlib.mlab import rec_join, rec2txt
from cvxopt import matrix, solvers
from cvxopt.blas import dot

def winsorize(array, percentile):
    assert percentile > 50, "percentile must be between 50 and 100"
    new_array = array.copy()
    upper = scoreatpercentile(array, percentile)
    lower = scoreatpercentile(array, 100 - percentile)
    new_array[np.where(array > upper)] = upper
    new_array[np.where(array < lower)] = lower
    return new_array


def stein_estimator_approx(scaled_t_stat_list):
    ## ignoring correlation structure
    ## scaled_t = x_bar / std_x_bar, has a std of 1
    assert len(scaled_t_stat_list) > 3
    y_norm2 = np.sum(np.array(scaled_t_stat_list) ** 2)
    sigma = 1.0
    est = [( 1 - (len(scaled_t_stat_list) - 2) * 1.0 / y_norm2 ) * st for st in scaled_t_stat_list]
    return est


def mean_estimation(list_of_arrays, use_stein=False):
    list_of_arrays = [winsorize(array, 99) for array in list_of_arrays]
    mean_of_arrays = [array.mean() for array in list_of_arrays]
    std_of_arrays = [array.std() for array in list_of_arrays]
    if use_stein:
        scaled_t_of_arrays = [mn / sd for (mn, sd) in zip(mean_of_arrays, std_of_arrays)]
        est = stein_estimator_approx(scaled_t_of_arrays)
        mean_of_arrays = [sd * stein for (sd, stein) in zip(std_of_arrays, est)]
    return mean_of_arrays

def cov_estimation(list_of_recarrays, index_name, pair_wise=False):
    def get_the_other_name(rec, index_name):
        assert len(rec.dtype.names) == 2
        name = [nm for nm in rec.dtype.names if nm != index_name]
        assert len(name) == 1
        return name[0]
    for array in list_of_recarrays:
        array[get_the_other_name(array, index_name)] = winsorize(array[get_the_other_name(array, index_name)])
    nn = len(list_of_recarrays)
    if not pair_wise:
        new_rec = list_of_recarrays[0]
        for ii in range(1, nn):
            new_rec = rec_join(index_name, new_rec, list_of_recarrays[ii], jointype='inner')
            dat_mat = np.c_[[new_rec[nm] for nm in new_rec.dtype.names if nm != index_name]].T
            covmat = np.cov(dat_mat)
    else :
        covmat = np.zeros((nn, nn))
        for ii in range(0, nn):
            covmat[ii,ii] = list_of_recarrays[ii][get_the_other_name(list_of_recarrays[ii], index_name)].var()
            for jj in range(ii+1, nn):
                new_rec = rec_join(index_name, list_of_recarrays[ii], list_of_recarrays[jj], jointype='inner')
                dat_mat = np.c_[[new_rec[nm] for nm in new_rec.dtype.names if nm != index_name]].T
                tmp_cov = np.cov(dat_mat)[0,1]
                covmat[ii,jj] = tmp_cov
                covmat[jj,ii] = tmp_cov
    return covmat

class MeanVariance(object):
    def __init__(self):
        """
        http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#quadratic-programming
        """
        self.nameList = []
        self.meanList = []
        self.covMat = None
        self.targetRisk = None
        self.optimizationMode = None

        self.optimizedWeights = []
        self.optimizedMean = None
        self.optimizedRisk = None


    def _optimize_with_no_constraint(self):
        ### common set up
        n = len(self.meanList)
        P = matrix(self.covMat)
        pbar = matrix(self.meanList)
        G = matrix(0.0, (n, n))
        h = matrix(0.0, (n, 1))
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        
        NN = 100
        mus = [ 10 ** (5.0 * t/NN - 1.0) for t in range(NN)]
        portfolios = [solvers.qp(mu * P, -pbar, G, h, A, b)['x'] for mu in mus]
        returns = [ dot(pbar, x) for x in portfolios ]
        risks = [np.sqrt(dot(x, P * x)) for x in portfolios]
        idx = np.argmin(np.abs(np.array(risks) - self.targetRisk))
        self.optimizedWeights = np.array(portfolios[idx]).T.tolist()[0]
        self.optimizedMean = returns[idx]
        self.optimizedRisk = risks[idx]
        
        

    def _optimize_with_positive_constraint(self):
        ### common set up
        n = len(self.meanList)
        P = matrix(self.covMat)
        pbar = matrix(self.meanList)
        G = matrix(0.0, (n, n))
        G[::n+1] = -1.0
        h = matrix(0.0, (n, 1))
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        
        NN = 100
        mus = [ 10 ** (5.0 * t/NN - 1.0) for t in range(NN)]
        portfolios = [solvers.qp(mu * P, -pbar, G, h, A, b)['x'] for mu in mus]
        returns = [ dot(pbar, x) for x in portfolios ]
        risks = [np.sqrt(dot(x, P * x)) for x in portfolios]
        idx = np.argmin(np.abs(np.array(risks) - self.targetRisk))
        self.optimizedWeights = np.array(portfolios[idx]).T.tolist()[0]
        self.optimizedMean = returns[idx]
        self.optimizedRisk = risks[idx]

    def _optimize_with_zeroone_constraint(self, normalize=True):
        ### common set up
        n = len(self.meanList)
        PP = matrix(self.covMat)
        P = self.covMat.copy()
        UU, ss, VV = np.linalg.svd(P)
        lamb = min(ss) * 0.99
        print lamb
        P[::n+1] -= lamb
        P = matrix(P)
        pbar = matrix(self.meanList)
        G = matrix(0.0, (n, n))
        G[::n+1] = -1.0
        h = matrix(0.0, (n, 1))
        A = matrix(1.0, (1, n))

        
        NN = 100
        mus = [ 10 ** (5.0 * t/NN - 1.0) for t in range(NN)]
        if normalize:
            #portfolios = [matrix(np.round(solvers.qp(mu * P, -pbar, G, h, A, matrix(nn*1.0))['x'] * nn) * 1.0 / nn) for mu in mus for nn in range(1, n+1)]
            portfolios = [matrix(1.0 * (np.round(solvers.qp(mu * P, -pbar + lamb, G, h, A, matrix(nn*1.0))['x']) > 0)) for mu in mus for nn in range(1, n+1)]
            portfolios = [x / sum(x) for x in portfolios]
        else :
            portfolios = [matrix(1.0 * (np.round(solvers.qp(mu * P, -pbar + lamb, G, h, A, matrix(nn*1.0))['x']) > 0)) for mu in mus for nn in range(1, n+1)]
            
        returns = [ dot(pbar, x) for x in portfolios ]
        risks = [np.sqrt(dot(x, PP * x)) for x in portfolios]
        idx = np.argmin(np.abs(np.array(risks) - self.targetRisk))
        self.optimizedWeights = np.array(portfolios[idx]).T.tolist()[0]
        self.optimizedMean = returns[idx]
        self.optimizedRisk = risks[idx]


    def run(self, mode, normalize=True):
        self.optimizationMode = mode
        if mode == 'no':
            self._optimize_with_no_constraint()
        elif mode == 'positive':
            self._optimize_with_positive_constraint()
        elif mode == 'zeroone':
            self._optimize_with_zeroone_constraint(normalize)


    def printResult(self):
        string = "\n"
        string+= "MeanVariance Optimization Result\n"
        string+= "Optimization Mode  : %s\n"%self.optimizationMode
        weight_data = np.array(zip(self.nameList, self.optimizedWeights), dtype=[('Name', 'S20'), ('Weight', float)])
        string+= "\n"
        string+= "Optimization Result:\n"
        string+= rec2txt(weight_data)
        string+= "\n"
        string+= "Optimization Metric:\n"
        string+= "                 Mean:  %f\n"%self.optimizedMean
        string+= "      (optimized)Risk:  %f\n"%self.optimizedRisk
        string+= "       (targeted)Risk:  %f\n"%self.targetRisk
        string+= "               Sharpe:  %f\n"%(self.optimizedMean / self.optimizedRisk)
        string+= "\n"
        return string


def test_mean_variance():
    mv = MeanVariance()
    mv.nameList = ['a', 'b', 'c']
    mv.meanList = [1.0, 1.0, 0.8]
    mv.covMat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    mv.targetRisk = 3.0
    mv.run('zeroone', True)
    print mv.printResult()

    
