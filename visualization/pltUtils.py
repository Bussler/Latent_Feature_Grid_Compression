from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def generate_array_MLFlow(client, idList, searchedParam):

    data = []
    realCompRates = []

    for id in idList:
        runs = client.search_runs(experiment_ids=id)
        bestrun = None
        for entry in runs:
            if bestrun is None or entry.data.params[searchedParam] < bestrun.data.params[searchedParam]:
                bestrun = entry
        data.append(float(bestrun.data.params[searchedParam]))
        realCompRates.append(float(bestrun.data.params['compression_ratio']))

    return data, realCompRates


def dict_from_file(filename):
    file = open(filename, 'r')
    Lines = file.readlines()

    d = {}
    for line in Lines:
        line = line.replace(' ', '')
        line = line.replace('\n', '')
        lineParts = line.split('=')

        value = lineParts[1]

        # M: parse int, float, list or string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if ',' in value:
                    value = value.replace('[', '')
                    value = value.replace(']', '')
                    value = value.split(',')

                    try:
                        value = [int(x) for x in value]
                    except ValueError:
                        value = [float(x) for x in value]
                else:
                    if value == 'True' or value == 'False':
                        value = bool(value)
                    else:  # M: normal string
                        value = lineParts[1]

        d[lineParts[0]] = value

    return d


def append_lists_from_dicts(lists: [], dict: {}, keys: []):
    if len(lists) is not len(keys):
        print("Error filling lists")

    for i, key in enumerate(keys):
        lists[i].append(dict[key])


def generate_plot_lists(lists: ([]), keys: ([]), BASENAME, config_names: (), experiment_names: []):

    if len(lists) is not len(keys) or len(lists) is not len(config_names):
        print("Error filling lists")

    for compr in experiment_names:
        for i in range(len(lists)):
            config_name = BASENAME + str(compr) + '/' + config_names[i]
            info = dict_from_file(config_name)
            append_lists_from_dicts(lists[i], info, keys[i])


def normalize_array_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize_array(array, minV, maxV, minN, maxN):
    return (maxN-minN) * ((array - minV) / (maxV - minV)) + minN


def generate_orderedValues(data, highestValue):
    steps = highestValue/len(data)
    np_array = np.array(data)
    data_sorted = np.argsort(np_array)

    for sort, index in enumerate(data_sorted):
        np_array[index] = steps * (sort+1)

    return np_array


def generateMeanValues(originalData, numbersOfElements):
    data_sum = []

    for i in range(0, len(originalData) - (numbersOfElements -1), numbersOfElements):
        cur = 0
        for j in range(numbersOfElements):
            cur += originalData[i+j]
        data_sum.append(cur/numbersOfElements)

    return data_sum


def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    return pareto_front

def findParetoValues(Xs, Ys, BASENAME, experimentNames, maxX=True, maxY=True):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    pf_X = [pair[0] for pair in pareto_front]

    infoName = 'info.txt'
    configName = 'config.txt'

    paretoConfigs = []

    for c in pf_X:
        for eN in experimentNames:
            foldername = BASENAME + str(eN)
            cName = foldername + '/'+infoName

            info = dict_from_file(cName)
            if info['compression_ratio'] == c:
                config = dict_from_file(foldername+'/'+configName)
                print(eN,': ', c, config['lr'], config['grad_lambda'], config['n_layers'])

                pc = [c, config['lr'], config['grad_lambda'], config['n_layers']]
                paretoConfigs.append(pc)

    return paretoConfigs