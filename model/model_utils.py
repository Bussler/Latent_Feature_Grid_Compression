import os


def write_dict(dictionary, filename, experiment_path=''):
    with open(os.path.join(experiment_path, filename), 'w') as f:
        for key, value in dictionary.items():
            f.write('%s = %s\n' % (key, value))