from training.training import training
from Feature_Grid_Training import config_parser
import numpy as np
import visualization.pltUtils as pu


def neurcompRunsDiffComprRatesFromFrontier():
    #BASENAME = 'experiments/NAS/mhd_p_baseline/mhd_p_'
    #experimentNames = np.linspace(0, 49, 50, dtype=int)
    #new_base_dir = '/experiments/Test_Diff_Wavelets/Debauchie_4/'

    BASENAME = 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_'
    experimentNames = np.linspace(0, 44, 45, dtype=int)
    new_base_dir = '/experiments/WithoutWaveletDecomp/mhd_p_Var_Static_ParetoFrontier/'

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    configs = pu.findParetoValues(CompressionRatio, PSNR, BASENAME, experimentNames, limitCompressionRatio=1000)

    for c in configs:

        c['config'] = ''
        c['Tensorboard_log_dir'] = ''
        c['basedir'] = new_base_dir

        #c['wavelet_filter'] = 'haar' # 'bior4.4' # 'db4'

        #c['expname'] = BASEEXPNAME + str(int(c[0]))
        c['checkpoint_path'] = ''
        c['feature_list'] = None
        training(c)


def fvRunsDiffComprRates():
    configName = 'experiment-config-files/test.txt'
    config = pu.dict_from_file(configName)

    BASEEXPNAME = '/experiments/QualityControl/LinearControl/Binary_Drop/'

    def simple_exponential_psigma(x):
        return 3.147258777702094 * x + np.log(1.0422820344614612e-10)

    def simple_exponential_dkl_mult(x):
        return 2.831272847362957 * x + np.log(6.692615916328201e-12)

    def simple_exponential_betas(x):
        return 1.5447992921914704 * x + np.log(9.345717687959087e-13)

    def simple_exponential_weights(x):
        return 5.0057915501801755 * x + np.log(9.71118711699754e-19)

    for compr in [100, 200, 300, 400, 500, 600]:

        #psigma = simple_exponential_psigma(np.log(compr))
        #dkl_mult = np.exp(simple_exponential_dkl_mult(np.log(compr)))
        beta = np.exp(simple_exponential_betas(np.log(compr)))
        weight = np.exp(simple_exponential_weights(np.log(compr)))

        #dkl_mult = np.exp(simple_exponential_dkl_multiplier(np.log(compr)))

        #print('Compr: ', compr, ' dkl_mult: ', dkl_mult)

        for i in range(1):

            config['basedir'] = BASEEXPNAME
            config['Tensorboard_log_dir'] = ''
            config['checkpoint_path'] = ''
            config['feature_list'] = None

            config['expname'] = 'mhd_p_' + str(compr) + '_' + str(i)

            # M: changing for compr
            #config['variational_sigma'] = psigma
            #config['weight_dkl_multiplier'] = dkl_mult
            config['lambda_drop_loss'] = beta
            config['lambda_weight_loss'] = weight

            training(config)


if __name__ == '__main__':
    neurcompRunsDiffComprRatesFromFrontier()
    #fvRunsDiffComprRates()
