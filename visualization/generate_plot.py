from cProfile import label

from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pltUtils import generate_array_MLFlow, dict_from_file, append_lists_from_dicts, generate_plot_lists,\
    normalize_array_0_1, normalize_array, generate_orderedValues, generateMeanValues, plot_pareto_frontier
import numpy as np
from itertools import product


def generateParetoFrontier():
    BASENAME = 'experiments/NAS/mhd_p_Smallify/mhd_p_'
    experimentNames = np.linspace(0, 79, 80, dtype=int)
    #experimentNames = np.delete(experimentNames, 5, axis=0)
    #experimentNames = np.delete(experimentNames, 5, axis=0)

    BASENAMEOther = 'experiments/NAS/mhd_p_Smallify_WithFinetuning/mhd_p_'
    experimentNamesOther = np.linspace(0, 49, 50, dtype=int)

    BASENAMEUnpruned = 'experiments/NAS/mhd_p_baseline/mhd_p_'
    experimentNamesUnpruned = np.linspace(0, 49, 60, dtype=int)
    #experimentNamesUnpruned = [221, 227, 246, 299, 327, 393, 503, 628]
    #experimentNamesUnpruned = [122, 135, 157, 198, 225, 292, 386, 534, 602, 781, 984, 1087]

    #BASENAMEUnpruned = 'experiments/diff_comp_rates/mhd_p_Baselines/100/mhd_p_'
    #experimentNamesUnpruned = [102, 144, 166, 211, 253, 268, 283, 293, 325, 363, 414, 442, 474, 512, 617, 638,
    #                           797, 895]
    #experimentNamesUnpruned = [210, 225, 235, 244, 296, 388, 463, 546, 596, 770, 931, 1251]
    #BASENAMEUnpruned = 'experiments/diff_comp_rates/mhd_p_Baselines/100_ForVariational/mhd_p_'
    #experimentNamesUnpruned = [105, 194, 283, 303, 311, 371, 468, 511, 603, 715, 808, 945, 1354]


    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    PSNRFinetuning = []
    CompressionRatioFinetuning = []

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

    pareto_front = plot_pareto_frontier(CompressionRatio, PSNR)
    pareto_frontFinetuning = plot_pareto_frontier(CompressionRatioFinetuning, PSNRFinetuning)
    pareto_frontUnpruned = plot_pareto_frontier(CompressionRatioUnpruned, PSNRUnpruned)

    '''Plotting process'''
    #plt.scatter(CompressionRatio, PSNR)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    pf_XFinetuning = [pair[0] for pair in pareto_frontFinetuning]
    pf_YFinetuning = [pair[1] for pair in pareto_frontFinetuning]

    pf_XUnpruned = [pair[0] for pair in pareto_frontUnpruned]
    pf_YUnpruned = [pair[1] for pair in pareto_frontUnpruned]

    limit = 1200

    newCompr = []
    newPSNR = []
    for i, k in zip(CompressionRatio, PSNR):
        if i < limit:
            newCompr.append(i)
            newPSNR.append(k)

    new_pf_X = []
    new_pf_Y = []
    for i,k in zip(pf_X, pf_Y):
        if i < limit:
            new_pf_X.append(i)
            new_pf_Y.append(k)

    new_pf_XFinetuning = []
    new_pf_YFinetuning = []
    for i, k in zip(pf_XFinetuning, pf_YFinetuning):
        if i < limit:
            new_pf_XFinetuning.append(i)
            new_pf_YFinetuning.append(k)

    new_pf_XUnpruned = []
    new_pf_YUnpruned = []
    for i, k in zip(pf_XUnpruned, pf_YUnpruned):
        if i < limit:
            new_pf_XUnpruned.append(i)
            new_pf_YUnpruned.append(k)

    plt.plot(new_pf_X, new_pf_Y, label='Pareto_Frontier Pruned', color='green')
    plt.plot(new_pf_XFinetuning, new_pf_YFinetuning, label='Pareto_Frontier Pruned With Finetuning', color='blue')
    #plt.scatter(newCompr, newPSNR, color='green', alpha =0.2)
    plt.plot(new_pf_XUnpruned, new_pf_YUnpruned, label='Baseline Unpruned', color='red')

    plt.xlabel('Compression_Ratio')
    plt.ylabel('PSNR')
    plt.legend()

    #print('Pareto-Compressionrates:')
    #for p in pf_X:
    #    print(p)

    filepath = 'plots/' + 'mhd_p_' + 'Smallify_Finetuning' + '.png'
    plt.savefig(filepath)


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
