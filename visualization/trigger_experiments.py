from training.training import training
from Feature_Grid_Training import config_parser
import numpy as np
import visualization.pltUtils as pu


def neurcompRunsDiffComprRatesFromFrontier():
    BASENAME = 'experiments/NAS/mhd_p_Smallify/mhd_p_'
    experimentNames = np.linspace(0, 79, 80, dtype=int)

    new_base_dir = '/experiments/NAS/mhd_p_Smallify_Pateto_Finetuning/'

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    configs = pu.findParetoValues(CompressionRatio, PSNR, BASENAME, experimentNames)

    for c in configs:

        c['config'] = ''
        c['Tensorboard_log_dir'] = ''
        c['basedir'] = new_base_dir

        #c['expname'] = BASEEXPNAME + str(int(c[0]))
        c['checkpoint_path'] = ''
        c['feature_list'] = None
        training(c)


if __name__ == '__main__':
    neurcompRunsDiffComprRatesFromFrontier()
