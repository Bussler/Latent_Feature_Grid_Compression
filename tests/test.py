import numpy as np
import torch
import pywt
import pywt.data
import matplotlib.pyplot as plt
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.model_utils import setup_model
from visualization.pltUtils import dict_from_file
import visualization.pltUtils as pu

def test_pywavelets():
    size_tensor = (32,64,64,64)

    feature_grid = torch.empty(size_tensor).uniform_(0, 1)

    # M: Testdata
    original = pywt.data.camera()  # M: ndarray

    # Method 1): Load image, Wavelet transform of image
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2

    # M: Method 2):
    coeffs = pywt.wavedec2(original, wavelet='db1', level=2)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # M: reconstruct
    #coeff_arr[0] = 0  #M: some filter
    coeffs_filt = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    reconstruction = pywt.waverec2(coeffs_filt, wavelet='db1')

    pass

def analyse_Wavelet():
    wavelet = pywt.Wavelet('db2')
    print(wavelet)

def test_TensorWavelets():
    size_tensor = (5, 16, 16, 16)
    feature_grid_orig = torch.empty(size_tensor).uniform_(0, 1)

    filter = WaveletFilter3d('db2')

    t = feature_grid_orig.shape[-3:]

    num_levels = None
    if num_levels is None:
        num_levels = min(pywt.dwt_max_level(s, filter.filter_length) for s in feature_grid_orig.shape[-3:])

    features = []  # M: wavelet coeffs representation of input feature grid
    shapes = []
    data = feature_grid_orig.detach().unsqueeze(0)
    for _ in range(num_levels):
        filtered, shape = filter.encode(data)
        features.append(filtered[0, :, 1:])
        shapes.append(shape)
        data = filtered[:, :, 0]  # M: only transform the lower part again

    features=[data[0]] + [*reversed(features)]
    shape_array = np.asarray(shapes[::-1], dtype=int)
    used_features = list([p for p in features])

    def decode_volume() -> torch.Tensor:
        restored = used_features[0].unsqueeze(0)
        for high_freq, shape in zip(used_features[1:], shape_array):
            data = torch.cat([restored.unsqueeze(2), high_freq.unsqueeze(0)], dim=2)
            restored = filter.decode(data, shape)
        return restored[0]

    decoded_feature_grid = decode_volume()
    pass


def analyse_coefficients():
    checkpoint_path = 'experiments/Tests/ImplTests/Finetuning/Test/model.pth'
    config_path = 'experiments/Tests/ImplTests/Finetuning/Test/config.txt'

    args = dict_from_file(config_path)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], '', args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])

    model.load_state_dict(torch.load(checkpoint_path))

    layers = []
    for name, layer in model.named_parameters():
        if 'feature_grid' in name:
            layers.append(layer.data.reshape(-1).cpu().numpy())

    fig, ax = plt.subplots(nrows=len(layers), ncols=1, figsize=(8, 8))
    fig.tight_layout()

    for i in range(len(layers)):
        ax[i].hist(layers[i], bins=160, label=str(i), range=(-0.5, 0.5))
        ax[i].title.set_text('Level ' + str(i))

    filepath = 'plots/Histogramms/' + 'test' + 'Feature_Historgramm_' + "Smallify" + '.png'
    plt.savefig(filepath)

def analyse_paretor_frontier():
    #BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Smallify/mhd_p_'
    BASENAME = 'experiments/NAS/mhd_p_baseline/mhd_p_'
    experimentNames = np.linspace(0, 49, 50, dtype=int)
    #experimentNames = [1,6,19,40,41,42,43,44]#[4,6,10,12,27,29,35,40,43,48,49,50,51,52]

    new_base_dir = '/experiments/NAS/mhd_p_MaskStraightThrough_Pateto_Finetuning/'

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                           (['psnr', 'compression_ratio'],),
                           BASENAME, (InfoName,), experiment_names=experimentNames)

    configs = pu.findParetoValues(CompressionRatio, PSNR, BASENAME, experimentNames)
    pass


def create_parallel_coordinates():
    import ast

    filename = 'experiments/Test_DiffDropratesPerLayer/Unpruned_Net_TestSet_WithEntropy/Results.txt'
    file = open(filename, 'r')
    Lines = file.readlines()

    # create data
    data = []
    d1 = []
    d2 = []
    d3 = []
    psnr = []
    compr = []
    x= np.linspace(0, 999, 1000, dtype=int)
    for line in Lines:
        d = ast.literal_eval(line)
        data.append(d)
        d1.append(d['pruning_threshold_list'][0])
        d2.append(d['pruning_threshold_list'][1])
        d3.append(d['pruning_threshold_list'][2])
        psnr.append(d['psnr'])
        compr.append(d['compression_ratio'])

    df = {'id': x,
          'Threshold Layer 1': d1,
          'Threshold Layer 2': d2,
          'Threshold Layer 3': d3,
          'PSNR': psnr,
          'Compression Ratio': compr}

    filename = 'plots/LatexFigures/Var_Droprate_Analysis/ParallelCoordPlots/testvol_Unpruned_WithEntropy_Parallel_Coordinates_ConstrainCompr'
    pu.generate_Parallel_Coordinate_Plot(df, filename, None, None)


def HyperparamNWWeights():
    BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_' #'experiments/NAS/mhd_p_Smallify/mhd_p_'
    experimentNames = np.linspace(0, 52, 53, dtype=int)
    # experimentNamesOther = [14,16,31,34,36,37,39,42,50,53,55,56,58,59,63,64,66,67,70,75,76]

    #BASENAME = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SearchNWArchitecture/mhd_p_'  # 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'
    #experimentNames = np.linspace(0, 79, 80, dtype=int)  # np.linspace(0, 54, 55, dtype=int)

    #BASENAME = 'experiments/NAS/mhd_p_Variational_Static_WithFinetuning_Buggy/mhd_p_'  # 'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_'
    #experimentNames = np.linspace(0, 79, 80, dtype=int)  # np.linspace(0, 44, 45, dtype=int)

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    pareto_front = pu.plot_pareto_frontier(CompressionRatio, PSNR)

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
                #print(config['expname'])

                #pc = [c, config['lambda_betas'], config['lambda_weights'], config['lr'], config['grad_lambda'], config['n_layers'], config['lr_decay']]
                paretoCompr.append(c)
                paretoPsnr.append(p)
                paretoLBeta.append(config['lambda_drop_loss'])
                paretoLWeight.append(config['lambda_weight_loss'])
                paretoMomentum.append(config['drop_momentum'])
                paretoThreshold.append(config['drop_threshold'])

                #paretoDKLMult.append(config['weight_dkl_multiplier'])
                #paretoSigma.append(config['variational_sigma'])

    plt.plot(paretoLBeta, paretoCompr)

    filepath = 'plots/' + 'test'
    plt.savefig(filepath + '.png')
    #tikzplotlib.save(filepath + '.pgf')
    pass


def fvRunsDiffComprRates():
    configName = 'experiment-config-files/mhd_p_basic.txt'
    config = dict_from_file(configName)

    BASEEXPNAME = '/experiments/QualityControl/LinearControl/Smallify/'

    def simple_exponential_dklMult(x):
        return -119.50757 * np.power(x, -1.46182)

    def simple_exponential_psigma(x):
        return -228.74157 * np.power(x, -0.67691)

    def simple_exponential_betas(x):
        return 2.2983364122806407 * x + np.log(1.1925636433232786e-14)

    def simple_exponential_weights(x):
        return 4.225962455634267 * x + np.log(4.613724748521028e-17)

    for compr in [100, 200, 300, 400, 500, 600]:

        #dkl_mult = np.exp(simple_exponential_dklMult(np.log(compr)))
        betas = np.exp(simple_exponential_betas(np.log(compr)))
        weights = np.exp(simple_exponential_weights(np.log(compr)))

        #psigma = simple_exponential_psigma(compr)

        print('Compr: ', compr, ' betas: ', betas, ' weights: ', weights)#, ' thresh: ', np.exp(thresh))


def RatioPruned_With_WithoutWavelets():
    from tests.CurveFitting import get_pareto_data
    import tikzplotlib

    BASENAME = 'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'#'experiments/NAS/mhd_p_Variational_Static_SetNWArchitecture/mhd_p_Variational_Static_SetNWArchitecturemhd_p_' #'experiments/NAS/mhd_p_Variational_Dynamic_WithFinetuning_SetNWArchitecture/mhd_p_'
    experimentNames = np.linspace(0, 57, 58, dtype=int)

    BASENAMEOther = 'experiments/WithoutWaveletDecomp/Var_Dynamic_2/mhd_p_'
    experimentNamesOther = [2,19,28,30,32,48,49,51,52,53,56,57]#[1, 6, 19, 40, 41, 42, 43, 44]  # [4,6,10,12,27,29,35,40,43,48,49,50,51,52]

    #BASENAME = 'experiments/NAS/mhd_p_Smallify_WithFinetuning_SetNWArchitecture/mhd_p_'
    #experimentNames = np.linspace(0, 52, 53, dtype=int)

    #BASENAMEOther = 'experiments/WithoutWaveletDecomp/Smallify_2/mhd_p_'
    #experimentNamesOther = [4,6,10,12,27,29,35,40,43,48,49,50,51,52]

    InfoName = 'info.txt'
    configName = 'config.txt'

    configs_WithWavelet, info_WithWavelet = get_pareto_data(BASENAME, experimentNames)

    configs_WithoutWavelet, info_WithoutWavelet = get_pareto_data(BASENAMEOther, experimentNamesOther)

    compr_with_Wavelet = []
    compr_without_Wavelet = []

    upper_limit = 600

    percentage_pruned_WithWavelet = []
    for entry in info_WithWavelet:
        if entry['compression_ratio'] < upper_limit:
            percentage_pruned_WithWavelet.append((entry['num_zeros'] / entry['num_parameters']) * 100.0)
            compr_with_Wavelet.append(entry['compression_ratio'])

    percentage_pruned_WithoutWavelet = []
    for entry in info_WithoutWavelet:
        if entry['compression_ratio'] < upper_limit:
            percentage_pruned_WithoutWavelet.append((entry['num_zeros'] / entry['num_parameters']) * 100.0)
            compr_without_Wavelet.append(entry['compression_ratio'])

    plt.plot(compr_with_Wavelet, percentage_pruned_WithWavelet, label='With Wavelet')
    plt.plot(compr_without_Wavelet, percentage_pruned_WithoutWavelet, label='Without Wavelet')

    plt.xlabel('Compression Ratio')
    plt.ylabel('Pruned in %')
    plt.legend()

    filepath = 'plots/LatexFigures/WaveletNoWavelet/Variational_Dynamic_PercentilePruned'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')

    pass


def test_model_storing():
    from model.model_utils import store_model_parameters

    checkpoint_path = 'experiments/Tests/turbulence/basic/model.pth'
    config_path = 'experiments/Tests/turbulence/basic/config.txt'

    args = dict_from_file(config_path)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], '', args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], checkpoint_path)

    store_model_parameters(model, "test_model_file")


def test_model_reading():
    from model.model_utils import restore_model
    from visualization.OutputToVTK import tiled_net_out
    from data.IndexDataset import get_tensor, IndexDataset

    filename = "test_model_file"
    config_path = 'experiments/Tests/turbulence/basic/config.txt'

    args = dict_from_file(config_path)

    model = restore_model(filename)

    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=True)


def test_binary_writing():
    test_string = '10010011'
    test_string_2 = '10100011010'

    n_bytes = len(test_string) // 8
    the_bytes = bytearray()
    for b in range(n_bytes):
        bin_val = test_string[8*b:8*b+8] #test_string[8*b:] if b==(n_bytes-1) else
        the_bytes.append(int(bin_val,2))

    with open("test_mask.bnr", "wb") as f:
        f.write(the_bytes)

        if len(test_string) % 8 != 0:
            bin_val = test_string[(8*n_bytes):]
            bin_val = bin_val + '0' * (8 - len(bin_val))
            barr = bytearray([int(bin_val, 2)])
            f.write(barr)

    pass


def test_read_binary():
    with open("test_mask.bnr", "rb") as file:
        test_string = '100100110001'
        b_size = (len(test_string) // 8) + 1 if len(test_string) % 8 != 0 else len(test_string) // 8

        inds = file.read(b_size)
        bits = ''.join(format(byte, '0' + str(8) + 'b') for byte in inds)
        pass
    

def test_kmeans_writing():
    #print("Cuda: ", torch.cuda.is_available())
    import math
    import struct
    from sklearn.cluster import KMeans
    
    def kmeans_quantization(w,q):
        weight_feat = w
        kmeans = KMeans(n_clusters=q,n_init=4).fit(weight_feat)
        return kmeans.labels_.tolist(),kmeans.cluster_centers_.reshape(q).tolist()
    
    def ints_to_bits_to_bytes(all_ints,n_bits):
        f_str = '#0'+str(n_bits+2)+'b'
        bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
        n_bytes = len(bit_string)//8
        the_leftover = len(bit_string)%8>0
        if the_leftover:
            n_bytes+=1
        the_bytes = bytearray()
        for b in range(n_bytes):
            bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
            the_bytes.append(int(bin_val,2))
        return the_bytes,the_leftover
    
    
    data = torch.randn(20, 20, 20)
    #print(data)
    
    filename = 'test_writing_kmeans'
    file = open(filename, 'wb')
    
    bit_precision = 9
    n_clusters = int(math.pow(2, bit_precision))
    
    # M: encode:
    weights_writing = 0
    
    weight_feat = data.view(-1).unsqueeze(1).numpy()
    
    labels, centers = kmeans_quantization(weight_feat, n_clusters)
    
    # weights
    w = centers
    w_format = ''.join(['f' for _ in range(len(w))])
    weights_writing += file.write(struct.pack(w_format, *w))
    weight_bin, is_leftover = ints_to_bits_to_bytes(labels, bit_precision)
    weights_writing += file.write(weight_bin)

    # encode non-pow-2 as 16-bit integer
    if bit_precision % 8 != 0:
        weights_writing += file.write(struct.pack('I', labels[-1]))
    
    file.flush()
    file.close()
    
    
    #! M: Opening again and reading
    file = open(filename,'rb')
    
    n_weights = data.numel()
    weight_size = (n_weights*bit_precision)//8
    if (n_weights*bit_precision)%8 != 0:
        weight_size+=1
    c_format = ''.join(['f' for _ in range(n_clusters)])
    centers = torch.FloatTensor(struct.unpack(c_format, file.read(4*n_clusters)))
    inds = file.read(weight_size)
    bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
    w_inds = torch.LongTensor([int(bits[bit_precision*i:bit_precision*i+bit_precision],2) for i in range(n_weights)])

    if bit_precision%8 != 0:
        next_bytes = file.read(4)
        w_inds[-1] = struct.unpack('I', next_bytes)[0]
        
        
    w_quant = centers[w_inds]
    all_ws=[w_quant]
    
    file.close()
    
    
    def calculate_deviation_statistics(prediction, ground_truth):
        diff_vol = ground_truth - prediction
        sqd_max_diff = (torch.max(ground_truth) - torch.min(ground_truth)) ** 2  # M: max für tthresh anpassen!!
        l1_diff = torch.mean(torch.abs(diff_vol))
        mse = torch.mean(torch.pow(diff_vol, 2.0))
        psnr = 10 * torch.log10(sqd_max_diff / mse)
        print('PSNR:', psnr, 'l1:', l1_diff, 'mse:', mse, 'rmse:', torch.sqrt(mse))
        return psnr.item(), l1_diff.item(), mse.item(), torch.sqrt(mse).item()
    
    calculate_deviation_statistics(w_quant.view(data.shape), data)
    
    pass


if __name__ == '__main__':
    #test_pywavelets()
    #analyse_Wavelet()
    #test_TensorWavelets()
    #analyse_coefficients()
    #analyse_paretor_frontier()
    #HyperparamNWWeights()
    #fvRunsDiffComprRates()

    #RatioPruned_With_WithoutWavelets()

    test_model_storing()
    #test_model_reading()
    
    #test_binary_writing()
    #test_read_binary()
    
    #test_kmeans_writing()