from training.training import training


if __name__ == '__main__':
    args={'data': 'datasets/mhd1024.h5',
          'd_in': 3,
          'd_out': 1,
          'sample_size': 16,
          'batch_size': 2048,
          'num_workers': 8,
          'max_pass': 20,
          'lr': 0.00017,
          'pass_decay': 20,
          'lr_decay': 0.2,
          'smallify_decay': 0,
          }
    training(args)
