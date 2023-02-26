import numpy as np
import visualization.pltUtils as pu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tikzplotlib


def get_pareto_data(BASENAME, experimentNames, InfoName='info.txt', configName='config.txt'):

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                           (['psnr', 'compression_ratio'],),
                           BASENAME, (InfoName,), experiment_names=experimentNames)

    pareto_front = pu.plot_pareto_frontier(CompressionRatio, PSNR)

    upper_limit = 1000
    pareto_configs = []

    for ppair in pareto_front:
        c = ppair[0]
        p = ppair[1]
        for eN in experimentNames:
            foldername = BASENAME + str(eN)
            cName = foldername + '/'+InfoName

            info = pu.dict_from_file(cName)
            if info['compression_ratio'] == c and c < upper_limit:
                config = pu.dict_from_file(foldername+'/'+configName)

                config['compression_ratio'] = c
                config['psnr'] = p

                pareto_configs.append(config)

    return pareto_configs


def split_data(configs:[]):

    paretoCompr = []
    paretoPsnr = []
    paretoLBeta = []
    paretoLWeight = []
    paretoMomentum = []
    paretoThreshold = []
    paretoDKLMult = []
    paretoSigma = []

    for entry in configs:
        paretoCompr.append(entry['compression_ratio'])
        paretoPsnr.append(entry['psnr'])
        paretoLBeta.append(entry['lambda_drop_loss'])
        paretoLWeight.append(entry['lambda_weight_loss'])
        paretoMomentum.append(entry['drop_momentum'])
        paretoThreshold.append(entry['drop_threshold'])
        paretoDKLMult.append(entry['weight_dkl_multiplier'])
        paretoSigma.append(entry['variational_sigma'])

    return {'compression_ratio' : paretoCompr,
            'psnr' : paretoPsnr,
            'lambda_drop_loss' : paretoLBeta,
            'lambda_weight_loss' : paretoLWeight,
            'drop_momentum' : paretoMomentum,
            'drop_threshold': paretoThreshold,
            'weight_dkl_multiplier' : paretoDKLMult,
            'variational_sigma' : paretoSigma}


def second_degree_polynomial(x, a, b, c):
    return a * x + b * x ** 2 + c


def second_degree_polynomial_MultiDim(X, a1, a2, b1, b2, c):
    x1, x2 = X
    return a1 * x1 + a2 * x2 + b1 * x1 ** 2 + b2 * x2 ** 2 + c


def Gauss(x, A, B, C):
    y = A*np.exp((((x-B)**2) / (2 * C**2)))
    return y


def third_degree_polynomial(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d


def fifth_degree_polynomial(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def simple_exponential(x, a, b):
    return a * np.power(x, b)


def fit_curve():
    # M: Get data
    #BASENAME = 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_'
    #experimentNames = np.linspace(0, 44, 45, dtype=int)

    #BASENAME = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'
    #experimentNames = np.linspace(0, 57, 58, dtype=int)

    #BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Smallify/mhd_p_'
    #experimentNames = np.linspace(0, 52, 53, dtype=int)

    BASENAME = 'experiments/NAS/mhd_p_MaskedStraightThrough_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNames = np.linspace(0, 54, 55, dtype=int)

    pareto_configs = get_pareto_data(BASENAME, experimentNames)
    data_list = split_data(pareto_configs)

    # M: curve fitting for linear model
    x_var = 'compression_ratio'#'psnr'
    #x_var2 = 'lambda_weight_loss'
    y_var = 'lambda_weight_loss'#'weight_dkl_multiplier'#'lambda_drop_loss'#'variational_sigma'

    x = np.log(np.asarray(data_list[x_var]))
    #x2 = np.asarray(data_list[x_var2])
    y = np.log(np.asarray(data_list[y_var]))

    #x = np.delete(x, 5, axis=0)
    #x2 = np.delete(x2, 5, axis=0)
    #y = np.delete(y, 5, axis=0)

    #X = (x, x2)

    for entry in zip(x, y):
        print(entry[0], entry[1])

    popt, _ = curve_fit(simple_exponential, x, y)
    #popt, _ = curve_fit(second_degree_polynomial, x, y)
    #popt, _ = curve_fit(second_degree_polynomial_MultiDim, X, y)
    #popt, _ = curve_fit(Gauss, x, y)
    #popt, _ = curve_fit(fifth_degree_polynomial, x, y)

    # M: Summarize Result
    a, b = popt
    print('y = %.5f * x ^ %.5f' % (a, b))
    #a, b, c = popt
    #print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    #a1, a2, b1, b2, c = popt
    #print(popt)
    #A, B, C = popt
    #print(popt)
    #a, b, c, d, e, f = popt
    #print('y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f * x^4 + %.5f * x^5 + %.5f' % (a, b, c, d, e, f))

    #ax = plt.gca()
    #ax.set(xscale = 'log', yscale = 'log')

    # M: Plotting
    plt.scatter(x, y, label='Baseline')  # M: GT

    x_line = np.linspace(min(x), max(x), 20, dtype=float)
    #x_Line2 = np.linspace(min(x2), max(x2), 20, dtype=float)
    #X_line = (x_line, x_Line2)

    y_line = simple_exponential(x_line, a, b)
    #y_line = second_degree_polynomial(x_line, a, b, c)
    #y_line = second_degree_polynomial_MultiDim(X_line, a1, a2, b1, b2, c)
    #y_line = Gauss(x_line, A, B, C)
    #y_line = fifth_degree_polynomial(x_line, a, b, c, d, e, f)

    plt.plot(x_line, y_line, '--', label='Fitted', color='crimson')

    plt.xlabel('log '+x_var)
    plt.ylabel('log '+y_var)
    plt.legend()

    filepath = 'plots/LatexFigures/AnalyseHyperparam/CurveFitting2/' + 'mhd_p_BinaryDrop_Weights_Exponential_SetArch.png'
    #filepath = 'plots/test'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')
    pass


if __name__ == '__main__':
    fit_curve()
