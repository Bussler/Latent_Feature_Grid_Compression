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
from data.IndexDataset import get_tensor, IndexDataset
from model.model_utils import setup_model


def generateParetoFrontier():
    BASENAME = 'experiments/NAS/mhd_p_MaskedStraightThrough_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_MaskStraightThrough/mhd_p_'
    experimentNames = np.linspace(0, 54, 55, dtype=int)

    BASENAMEOther = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Smallify/mhd_p_'
    experimentNamesOther = np.linspace(0, 52, 53, dtype=int)
    #experimentNamesOther = [14,16,31,34,36,37,39,42,50,53,55,56,58,59,63,64,66,67,70,75,76]

    BASENAMEOther2 = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SearchNWArchitecture/mhd_p_'
    experimentNamesOther2 = np.linspace(0, 57, 58, dtype=int)#np.linspace(0, 54, 55, dtype=int)

    BASENAMEOther3 = 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_' #'experiments/NAS/mhd_p_Variational_Static_WithFinetuning_Buggy/mhd_p_'
    experimentNamesOther3 = np.linspace(0, 44, 45, dtype=int)#np.linspace(0, 44, 45, dtype=int)

    BASENAMEUnpruned = 'experiments/NAS/mhd_p_baseline/mhd_p_'
    experimentNamesUnpruned = np.linspace(0, 49, 50, dtype=int)


    BASENAME = 'experiments/NAS_testVol/Binary_SearchNWArch/testvol_'#'experiments/NAS_testVol/Variational_Dynamic_SearchNWArch/testvol_' #'experiments/NAS_testVol/Smallify_SearchNWArch/testvol_'
    experimentNames = np.linspace(0, 69, 70, dtype=int)

    BASENAMEUnpruned = 'experiments/NAS_testVol/baseline/testvol_'
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

    plt.plot(new_pf_X, new_pf_Y, label='Pruned')
    plt.scatter(newCompr, newPSNR, label='Pruned', alpha =0.2)
    plt.plot(new_pf_XUnpruned, new_pf_YUnpruned, label='Baseline Unpruned')

    #plt.plot(new_pf_XFinetuning, new_pf_YFinetuning, label='Pareto Frontier Smallify')
    #plt.plot(new_pf_XOther2, new_pf_YOther2, label='Pareto Frontier Variational Dynamic')
    #plt.plot(new_pf_XOther3, new_pf_YOther3, label='Pareto Frontier Variational Static')

    plt.xlabel('Compression_Ratio')
    plt.ylabel('PSNR')
    plt.legend()

    #print('Pareto-Compressionrates:')
    #for p in pf_X:
    #    print(p)

    #filepath = 'plots/' + 'test'
    filepath = 'plots/LatexFigures/testvol_NAS/Binary'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')


def generateParetoFrontier_WithoutWavelet():
    BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNames = np.linspace(0, 52, 53, dtype=int)

    #BASENAME = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'
    #experimentNames = np.linspace(0, 57, 58, dtype=int)

    BASENAMEOther = 'experiments/WithoutWaveletDecomp/Smallify_2/mhd_p_'
    experimentNamesOther = [4,6,10,12,27,29,35,40,43,48,49,50,51,52]

    #BASENAMEOther = 'experiments/WithoutWaveletDecomp/Var_Static_2/mhd_p_'
    #experimentNamesOther = [1,6,19,40,41,42,43,44]#[2,19,28,30,32,48,49,51,52,53,56,57]

    BASENAMEOther2 = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SearchNWArchitecture/mhd_p_'
    experimentNamesOther2 = np.linspace(0, 57, 58, dtype=int)#np.linspace(0, 54, 55, dtype=int)

    BASENAMEOther3 = 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_' #'experiments/NAS/mhd_p_Variational_Static_WithFinetuning_Buggy/mhd_p_'
    experimentNamesOther3 = np.linspace(0, 44, 45, dtype=int)#np.linspace(0, 44, 45, dtype=int)

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

    plt.plot(new_pf_X, new_pf_Y, label='With Wavelet')
    #plt.scatter(newCompr, newPSNR, label='Pruned', alpha =0.2)
    #plt.plot(new_pf_XUnpruned, new_pf_YUnpruned, label='Baseline Unpruned')

    plt.plot(new_pf_XFinetuning, new_pf_YFinetuning, label='Without Wavelet')
    #plt.scatter(CompressionRatioFinetuning, PSNRFinetuning, label='Without Wavelet', alpha=0.2)
    #plt.plot(new_pf_XOther2, new_pf_YOther2, label='Pareto Frontier Variational Dynamic')
    #plt.plot(new_pf_XOther3, new_pf_YOther3, label='Pareto Frontier Variational Static')

    plt.xlabel('Compression Ratio')
    plt.ylabel('PSNR')
    plt.legend()

    #print('Pareto-Compressionrates:')
    #for p in pf_X:
    #    print(p)

    #filepath = 'plots/' + 'test_Smallify'
    filepath = 'plots/LatexFigures/WaveletNoWavelet/Smallify'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')


def HyperparamAnalysis():
    #BASENAME = 'experiments/NAS/mhd_p_MaskedStraightThrough_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_MaskStraightThrough/mhd_p_'
    #experimentNames = np.linspace(0, 54, 55, dtype=int)

    #BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Smallify/mhd_p_'
    #experimentNames = np.linspace(0, 52, 53, dtype=int)
    # experimentNamesOther = [14,16,31,34,36,37,39,42,50,53,55,56,58,59,63,64,66,67,70,75,76]

    #BASENAME = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SearchNWArchitecture/mhd_p_'
    #experimentNames = np.linspace(0, 57, 58, dtype=int)  # np.linspace(0, 54, 55, dtype=int)

    BASENAME = 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_'  # 'experiments/NAS/mhd_p_Variational_Static_WithFinetuning_Buggy/mhd_p_'
    experimentNames = np.linspace(0, 44, 45, dtype=int)  # np.linspace(0, 44, 45, dtype=int)

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
    paretoMomentum = []
    paretoThreshold = []

    paretoDKLMult = []
    paretoSigma = []

    for ppair in pareto_front:
        c = ppair[0]
        p = ppair[1]
        for eN in experimentNames:
            foldername = BASENAME + str(eN)
            cName = foldername + '/'+InfoName

            info = dict_from_file(cName)
            if info['compression_ratio'] == c and c < 1000:
                config = dict_from_file(foldername+'/'+configName)
                print(config['expname'])

                #pc = [c, config['lambda_betas'], config['lambda_weights'], config['lr'], config['grad_lambda'], config['n_layers'], config['lr_decay']]
                paretoCompr.append(c)
                paretoPsnr.append(p)
                paretoLBeta.append(config['lambda_drop_loss'])
                paretoLWeight.append(config['lambda_weight_loss'])
                paretoMomentum.append(config['drop_momentum'])
                paretoThreshold.append(config['drop_threshold'])

                paretoDKLMult.append(config['weight_dkl_multiplier'])
                paretoSigma.append(config['variational_sigma'])

    #prune = 5
    #paretoLBeta = np.delete(paretoLBeta, prune, axis=0)
    #paretoCompr = np.delete(paretoCompr, prune, axis=0)
    #paretoPsnr = np.delete(paretoPsnr, prune, axis=0)
    #paretoLWeight = np.delete(paretoLWeight, prune, axis=0)
    #paretoMomentum = np.delete(paretoMomentum, prune, axis=0)
    #paretoThreshold = np.delete(paretoThreshold, prune, axis=0)

    '''Plotting process'''
    #fig, (( ax3,ax4,ax5, ax6),( ax9, ax10,ax11, ax12)) = plt.subplots(2, 4, figsize=(13, 13), dpi= 200) #ax5, ax11,

    figure, axis = plt.subplots(2, 2, figsize=(7, 7))

    axis[0,0].plot(paretoDKLMult,paretoCompr, color = 'orange') #paretoLBeta
    axis[0,0].title.set_text('DKL Ramp Up')

    axis[0,1].plot(paretoSigma,paretoCompr, color = 'red') #paretoLWeight
    axis[0,1].title.set_text('PSigma')

    axis[1,0].plot(paretoMomentum,paretoCompr, color = 'violet')
    axis[1,0].title.set_text('Momentum')

    axis[1,1].plot(paretoThreshold,paretoCompr, color = 'black')
    axis[1,1].title.set_text('Pruning Threshold')

    axis[0, 0].set_ylabel('Compression Rate')
    axis[1, 0].set_ylabel('Compression Rate')

    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #axis[0,0].set_xticks(paretoDKLMult)
    #axis[0,0].set_xticklabels(['{:,.5f}'.format(num) for num in paretoDKLMult])
    axis[0,0].xaxis.set_major_formatter(FormatStrFormatter('%g'))

    filepath = 'plots/LatexFigures/AnalyseHyperparam/SetArch/BiggerFigure/' + 'mhd_p_Variational_Static_SetArch_HyperparamAnalyis'
    #filepath = 'plots/test'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')
    return


def WeightHistogramm():
    from model.Variational_Dropout_Layer import VariationalDropout
    from model.Smallify_Dropout import SmallifyDropout
    from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout
    from data.IndexDataset import get_tensor, IndexDataset
    from model.model_utils import setup_model
    import torch

    #Configpath = 'experiments/Test_DiffDropratesPerLayer/mhd_p_Variational_Static/Unpruned_Variational_Static_500/config.txt'
    #Configpath = 'experiments/Test_DiffDropratesPerLayer/mhd_p_Variational_Dynamic/Unpruned_Variational_Dynamic_500/config.txt'
    #Configpath = 'experiments/Test_DiffDropratesPerLayer/mhd_p_Smallify/Unpruned_Smallify_500/config.txt'
    Configpath = 'experiments/Test_DiffDropratesPerLayer/mhd_p_Masked/Unpruned_Masked_500/config.txt'
    args = dict_from_file(Configpath)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], args['drop_type'], args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])

    layers = []
    for layer in model.drop.modules():
        #if isinstance(layer, Linear):
        #    layers.append(layer.weight.data)
        if isinstance(layer, VariationalDropout):
            layers.append(layer.dropout_rates.detach().data.reshape(-1).numpy())
        if isinstance(layer, SmallifyDropout):
            layers.append(layer.betas.detach().data.reshape(-1).numpy())
        if isinstance(layer, MaskedWavelet_Straight_Through_Dropout):
            layers.append(torch.sigmoid(layer.mask_values).detach().data.reshape(-1).numpy())

    fig, ax = plt.subplots(nrows=len(layers), ncols=1, figsize=(8, 8))
    fig.tight_layout()

    for i in range(len(layers)):
        ax[i].hist(layers[i], bins=120, label=str(i))  # range=(-0.5, 0.5)
        ax[i].title.set_text('Layer '+str(i))

    filepath = 'plots/LatexFigures/AnalyseNWWeights/' + 'mhd_p_' + 'MaskedStraightThrough_500_' + 'Weight_Historgramm'
    plt.savefig(filepath + '.png')
    tikzplotlib.save(filepath + '.pgf')


def curve_quality_control_plot():
    BASENAME = 'experiments/QualityControl/LinearControl/Variational_Static/mhd_p_'

    comprnames = [100, 200, 300, 400, 500, 600]
    expnames = [0]

    InfoName = 'info.txt'
    configName = 'config.txt'

    metric_variational_sigma = 'variational_sigma'
    metric_weight_dkl_multiplier = 'weight_dkl_multiplier'
    metric_lambda_drop_loss = 'lambda_drop_loss'
    metric_lambda_weight_loss = 'lambda_weight_loss'

    metric_1_name = metric_variational_sigma
    comparison_1_name = 'compression_ratio'

    def simple_exponential_psigma(x):
        return 3.147258777702094 * x + np.log(1.0422820344614612e-10)

    def simple_exponential_dkl_mult(x):
        return 2.831272847362957 * x + np.log(6.692615916328201e-12)

    def simple_exponential_betas_smallify(x):
        return 2.2983364122806407 * x + np.log(1.1925636433232786e-14)

    def simple_exponential_weights_smallify(x):
        return 4.225962455634267 * x + np.log(4.613724748521028e-17)

    def simple_exponential_betas_binary(x):
        return 1.5447992921914704 * x + np.log(9.345717687959087e-13)

    def simple_exponential_weights_binary(x):
        return 5.0057915501801755 * x + np.log(9.71118711699754e-19)

    metric_1 = []
    comparison_1 = []

    for c in comprnames:

        for e in expnames:
            config = dict_from_file(BASENAME + str(c) + '_' + str(e) + '/' + configName)
            info = dict_from_file(BASENAME + str(c) + '_' + str(e) + '/' + InfoName)

            metric_1.append(config[metric_1_name])
            comparison_1.append(info[comparison_1_name])

    #ax = plt.gca()
    #ax.set(xscale = 'log', yscale = 'log')

    plt.scatter(np.log(comparison_1), (metric_1), alpha=0.5, label='Ground Truth Runs', color = 'steelblue')  # M: Experiments
    #x_line_scatter = np.linspace(min(comparison_1), max(comparison_1), len(metric_1), dtype=float)
    #plt.fill_between(comparison_1_values,min_metric_1, max_metric_1, alpha=0.5, color = 'steelblue') -> Unterscheiden sich doch nur in x!

    x_line = np.linspace(100, 800, 20, dtype=float)
    y_line_metric1 = (simple_exponential_psigma(np.log(x_line)))

    plt.plot(np.log(x_line), y_line_metric1, label='Fitted Curve', color = 'forestgreen')

    plt.xlabel('log '+comparison_1_name)
    plt.ylabel('log '+metric_1_name)
    plt.legend()

    filepath = 'plots/LatexFigures/AnalyseHyperparam/QualityControl3/' + 'mhd_p_Variational_Static'
    #filepath = 'test2'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')


if __name__ == '__main__':
    #generateParetoFrontier()
    #generateParetoFrontier_WithoutWavelet()
    #WeightHistogramm()
    HyperparamAnalysis()
    #curve_quality_control_plot()
