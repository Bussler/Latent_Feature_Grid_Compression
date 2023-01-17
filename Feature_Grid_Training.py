from training.training import training


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument("--expname", type=str, required=True,
                        help='name of your experiment; is required')
    parser.add_argument("--data", type=str, required=True,
                        help='path to volume data set; is required')
    parser.add_argument("--basedir", type=str, default='/experiments/',
                        help='where to store ckpts and logs')
    parser.add_argument("--Tensorboard_log_dir", type=str, default='',
                        help='where to store tensorboard logs')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--sample_size', type=int, default=16, help='how many indices to generate per batch item')
    parser.add_argument('--num_workers', type=int, default=8, help='how many parallel workers for batch access')

    parser.add_argument('--max_pass', type=int, default=75,
                        help='number of training passes to make over the volume, default=75')
    parser.add_argument('--lr', type=float, default=0.008, help='learning rate, default=0.008')
    parser.add_argument('--pass_decay', type=int, default=20,
                        help='training-pass-amount at which to decay learning rate, default=20')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')
    parser.add_argument('--smallify_decay', type=int, default=0,
                        help='Option to enable loss decay as presented in the smallify-paper:'
                             'Every smallify_decay - epochs without improvement, the learning rate'
                             'is divided by 10 up until 1e-7. Default: 0 to turn option off')

    parser.add_argument('--lambda_drop_loss', type=float, default=1.e-8, help='weighting of drop-loss')
    parser.add_argument('--lambda_weight_loss', type=float, default=1.e-8, help='weighting of weight-loss')

    parser.add_argument('--d_in', type=int, default=3, help='spatial input dimension')
    parser.add_argument('--d_out', type=int, default=1, help='spatial output dimension')
    parser.add_argument('--n_hidden_size', type=int, default=32, help='size of hidden layers in network')
    parser.add_argument('--n_layers', type=int, default=4, help='number of layers for the network')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint from where to load model')

    parser.add_argument('--embedding_type', type=str, default='fourier',
                        help='periodic functions to use for frequency embedding')
    parser.add_argument('--n_embedding_freq', type=int, default=2, help='number of frequency bands used for high'
                                                                        ' frequency embedding')
    parser.add_argument('--drop_type', type=str, default='smallify',
                        help='Type of dropout algorithm to use. Options are: <smallify>,')
    parser.add_argument('--wavelet_filter', type=str, default='db2', help='checkpoint from where to load model')
    parser.add_argument('--grid_features', type=int, default=16,
                        help='Amount of features at each point in feature grid')
    parser.add_argument('--grid_size', type=int, default=32,
                        help='Size of feature grid in x, y, z dimension')

    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())
    print("Finished parsing arguments, starting training")
    training(args)
