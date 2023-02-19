from cProfile import label

from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pltUtils import generate_array_MLFlow, dict_from_file, append_lists_from_dicts, generate_plot_lists,\
    normalize_array_0_1, normalize_array, generate_orderedValues, generateMeanValues, plot_pareto_frontier
import numpy as np
from itertools import product
import tikzplotlib


def generateParetoFrontier():
    BASENAME = 'experiments/NAS/mhd_p_MaskStraightThrough/mhd_p_'#'experiments/NAS/mhd_p_MaskedStraightThrough_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNames = np.linspace(0, 79, 80, dtype=int)#np.linspace(0, 49, 50, dtype=int)

    BASENAMEOther = 'experiments/NAS/mhd_p_Smallify/mhd_p_'#'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNamesOther = np.linspace(0, 79, 80, dtype=int)
    #experimentNamesOther = [14,16,31,34,36,37,39,42,50,53,55,56,58,59,63,64,66,67,70,75,76]

    BASENAMEOther2 = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SearchNWArchitecture/mhd_p_'#'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNamesOther2 = np.linspace(0, 79, 80, dtype=int)#np.linspace(0, 54, 55, dtype=int)

    BASENAMEOther3 = 'experiments/NAS/mhd_p_Variational_Static_WithFinetuning_Buggy/mhd_p_'#'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_'
    experimentNamesOther3 = np.linspace(0, 79, 80, dtype=int)#np.linspace(0, 44, 45, dtype=int)

    BASENAMEUnpruned = 'experiments/NAS/mhd_p_baseline/mhd_p_'
    experimentNamesUnpruned = np.linspace(0, 49, 50, dtype=int)

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    PSNRFinetuning = []
    CompressionRatioFinetuning = []

    PSNROther2 = []
    CompressionRatioOther2 = []

    PSNROther3 = []
    CompressionRatioOther3 = []

    PSNRUnpruned = []
    CompressionRatioUnpruned = []

    generate_plot_lists(([PSNR, CompressionRatio],),
                           (['psnr', 'compression_ratio'],),
                           BASENAME, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([PSNRFinetuning, CompressionRatioFinetuning],),
                        (['psnr', 'compression_ratio'],),
                        BASENAMEOther, (InfoName,), experiment_names=experimentNamesOther)

    generate_plot_lists(([PSNRUnpruned, CompressionRatioUnpruned],),
                        (['psnr', 'compression_ratio'],),
                        BASENAMEUnpruned, (InfoName,), experiment_names=experimentNamesUnpruned)

    generate_plot_lists(([PSNROther2, CompressionRatioOther2],),
                        (['psnr', 'compression_ratio'],),
                        BASENAMEOther2, (InfoName,), experiment_names=experimentNamesOther2)

    generate_plot_lists(([PSNROther3, CompressionRatioOther3],),
                        (['psnr', 'compression_ratio'],),
                        BASENAMEOther3, (InfoName,), experiment_names=experimentNamesOther3)

    pareto_front = plot_pareto_frontier(CompressionRatio, PSNR)
    pareto_frontFinetuning = plot_pareto_frontier(CompressionRatioFinetuning, PSNRFinetuning)
    pareto_frontUnpruned = plot_pareto_frontier(CompressionRatioUnpruned, PSNRUnpruned)
    pareto_frontOther2 = plot_pareto_frontier(CompressionRatioOther2, PSNROther2)
    pareto_frontOther3 = plot_pareto_frontier(CompressionRatioOther3, PSNROther3)

    '''Plotting process'''
    #plt.scatter(CompressionRatio, PSNR)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    pf_XFinetuning = [pair[0] for pair in pareto_frontFinetuning]
    pf_YFinetuning = [pair[1] for pair in pareto_frontFinetuning]

    pf_XUnpruned = [pair[0] for pair in pareto_frontUnpruned]
    pf_YUnpruned = [pair[1] for pair in pareto_frontUnpruned]

    pf_XOther2 = [pair[0] for pair in pareto_frontOther2]
    pf_YOther2 = [pair[1] for pair in pareto_frontOther2]

    pf_XOther3 = [pair[0] for pair in pareto_frontOther3]
    pf_YOther3 = [pair[1] for pair in pareto_frontOther3]

    upper_limit = 600
    lower_limit = 0

    newCompr = []
    newPSNR = []
    for i, k in zip(CompressionRatio, PSNR):
        if i < upper_limit and i > lower_limit:
            newCompr.append(i)
            newPSNR.append(k)

    new_pf_X = []
    new_pf_Y = []
    for i,k in zip(pf_X, pf_Y):
        if i < upper_limit and i > lower_limit:
            new_pf_X.append(i)
            new_pf_Y.append(k)

    new_pf_XFinetuning = []
    new_pf_YFinetuning = []
    for i, k in zip(pf_XFinetuning, pf_YFinetuning):
        if i < upper_limit and i > lower_limit:
            new_pf_XFinetuning.append(i)
            new_pf_YFinetuning.append(k)

    new_pf_XOther2 = []
    new_pf_YOther2 = []
    for i, k in zip(pf_XOther2, pf_YOther2):
        if i < upper_limit and i > lower_limit:
            new_pf_XOther2.append(i)
            new_pf_YOther2.append(k)

    new_pf_XOther3 = []
    new_pf_YOther3 = []
    for i, k in zip(pf_XOther3, pf_YOther3):
        if i < upper_limit and i > lower_limit:
            new_pf_XOther3.append(i)
            new_pf_YOther3.append(k)

    new_pf_XUnpruned = []
    new_pf_YUnpruned = []
    for i, k in zip(pf_XUnpruned, pf_YUnpruned):
        if i < upper_limit and i > lower_limit:
            new_pf_XUnpruned.append(i)
            new_pf_YUnpruned.append(k)

    plt.plot(new_pf_X, new_pf_Y, label='Pareto Frontier MaskedStraightThrough')
    #plt.scatter(newCompr, newPSNR, color='green', alpha =0.2)
    plt.plot(new_pf_XUnpruned, new_pf_YUnpruned, label='Baseline Unpruned')

    plt.plot(new_pf_XFinetuning, new_pf_YFinetuning, label='Pareto Frontier Smallify')
    plt.plot(new_pf_XOther2, new_pf_YOther2, label='Pareto Frontier Variational Dynamic')
    plt.plot(new_pf_XOther3, new_pf_YOther3, label='Pareto Frontier Variational Static')

    plt.xlabel('Compression_Ratio')
    plt.ylabel('PSNR')
    plt.legend()

    #print('Pareto-Compressionrates:')
    #for p in pf_X:
    #    print(p)

    #filepath = 'plots/' + 'mhd_p_' + 'SetNWArch_DropComparisons' + '.png'
    filepath = 'plots/LatexFigures/comprBaselines/mhd_p_DropComparison_SearchNWArch_Pruned_VS_Unpruned'
    plt.savefig(filepath + '.png')
    tikzplotlib.save(filepath + '.pgf')


def HyperparamAnalysis():
    BASENAME = 'experiments/hyperparam_search/mhd_p_NAS/200_WithFinetuning/mhd_p_200_'
    experimentNames = np.linspace(0, 49, 50, dtype=int)

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    pareto_front = plot_pareto_frontier(CompressionRatio, PSNR)

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    paretoCompr = []
    paretoPsnr = []
    paretoLBeta = []
    paretoLWeight = []
    paretoLR = []
    paretoLGrad = []
    paretoNLayer = []
    paretoLRDecay = []

    for ppair in pareto_front:
        c = ppair[0]
        p = ppair[1]
        for eN in experimentNames:
            foldername = BASENAME + str(eN)
            cName = foldername + '/'+InfoName

            info = dict_from_file(cName)
            if info['compression_ratio'] == c:
                config = dict_from_file(foldername+'/'+configName)

                #pc = [c, config['lambda_betas'], config['lambda_weights'], config['lr'], config['grad_lambda'], config['n_layers'], config['lr_decay']]
                paretoCompr.append(c)
                paretoPsnr.append(p)
                paretoLBeta.append(config['lambda_betas'])
                paretoLWeight.append(config['lambda_weights'])
                paretoLR.append(config['lr'])
                paretoLGrad.append(config['grad_lambda'])
                paretoNLayer.append(config['n_layers'])
                paretoLRDecay.append(config['lr_decay'])

    '''Plotting process'''
    fig, ((ax1, ax2,ax3,ax4,ax5, ax6),(ax7,ax8,ax9, ax10,ax11,ax12)) = plt.subplots(2, 6, figsize=(15, 15), dpi= 100)

    ax1.plot(paretoLR,paretoCompr, label='LR', color = 'blue')
    ax1.title.set_text('LR')
    ax1.set_xlabel('lr')
    ax1.set_ylabel('Compression Rate')

    ax7.plot(paretoLR,paretoPsnr, label='LR', color = 'blue')
    ax7.title.set_text('LR')
    ax7.set_xlabel('lr')
    ax7.set_ylabel('PSNR')

    ax2.plot(paretoLGrad,paretoCompr, label='Lambda Gradient', color = 'green')
    ax2.title.set_text('Lambda Gradient')
    ax2.set_xlabel('Lambda Gradient')
    ax2.set_ylabel('Compression Rate')

    ax8.plot(paretoLGrad,paretoPsnr, label='Lambda Gradient', color = 'green')
    ax8.title.set_text('Lambda Gradient')
    ax8.set_xlabel('Lambda Gradient')
    ax8.set_ylabel('PSNR')

    ax3.plot(paretoLBeta,paretoCompr, label='Lambda Beta', color = 'orange')
    ax3.title.set_text('Lambda Beta')
    ax3.set_xlabel('Lambda Beta')
    ax3.set_ylabel('Compression Rate')

    ax9.plot(paretoLBeta,paretoPsnr, label='Lambda Beta', color = 'orange')
    ax9.title.set_text('Lambda Beta')
    ax9.set_xlabel('Lambda Beta')
    ax9.set_ylabel('PSNR')

    ax4.plot(paretoLWeight,paretoCompr, label='Lambda Weight', color = 'red')
    ax4.title.set_text('Lambda Weight')
    ax4.set_xlabel('Lambda Weight')
    ax4.set_ylabel('Compression Rate')

    ax10.plot(paretoLWeight,paretoPsnr, label='Lambda Weight', color = 'red')
    ax10.title.set_text('Lambda Weight')
    ax10.set_xlabel('Lambda Weight')
    ax10.set_ylabel('PSNR')

    ax5.plot(paretoNLayer,paretoCompr, label='N Layer', color = 'violet')
    ax5.title.set_text('N Layer')
    ax5.set_xlabel('N Layer')
    ax5.set_ylabel('Compression Rate')

    ax11.plot(paretoNLayer,paretoPsnr, label='N Layer', color = 'violet')
    ax11.title.set_text('N Layer')
    ax11.set_xlabel('N Layer')
    ax11.set_ylabel('PSNR')

    ax6.plot(paretoLRDecay,paretoCompr, label='LR Decay', color = 'black')
    ax6.title.set_text('LR Decay')
    ax6.set_xlabel('LR Decay')
    ax6.set_ylabel('Compression Rate')

    ax12.plot(paretoLRDecay,paretoPsnr, label='LR Decay', color = 'black')
    ax12.title.set_text('LR Decay')
    ax12.set_xlabel('LR Decay')
    ax12.set_ylabel('PSNR')

    plt.legend()
    filepath = 'plots/' + 'mhd_p_' + "200_Finetuning_" + 'HyperparamAnalyis' + '.png'
    plt.savefig(filepath)


if __name__ == '__main__':
    generateParetoFrontier()
