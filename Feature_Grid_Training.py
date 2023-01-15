from training.training import training


if __name__ == '__main__':
    args={'expname': 'Test_NOGrid_WithFEmbedding',
          'basedir': '/experiments/Tests/ImplTests/',
          'data': 'datasets/test_vol.npy',
          'd_in': 3,
          'd_out': 1,
          'sample_size': 16,
          'batch_size': 1024,
          'num_workers': 8,
          'max_pass': 20,
          'lr': 0.00017,
          'pass_decay': 20,
          'lr_decay': 0.2,
          'smallify_decay': 0,
          'Tensorboard_log_dir': 'runs/Tests/implTestNW'
          }
    training(args)
