import json
import pickle as pkl
import sys

import numpy as np

# Порядок отведений
LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
DATA_FOLDER = 'data\\'
FREQUENCY_OF_DATASET = 250

OLD_DATASET_FOLDER = DATA_FOLDER + 'old\\'
NEW_DATASET_FOLDER = DATA_FOLDER + 'new\\'

RAW_OLD_DATASET_PATH = OLD_DATASET_FOLDER + "data_1078.json"
RAW_NEW_DATASET_PATH = NEW_DATASET_FOLDER + "data_2033.json"

OLD_AND_NEW_PKL_PATH = DATA_FOLDER + "dataset_fixed_baseline.pkl"
DIAGNOSIS_DICTIONARY = DATA_FOLDER + "diagnosis.json"

OLD_PKL_PATH = OLD_DATASET_FOLDER + "dataset_fixed_baseline.pkl"

OLD_6002_PATH = OLD_DATASET_FOLDER + "6002_norm_old.pkl"
NEW_6002_PATH = NEW_DATASET_FOLDER + "6002_norm_new.pkl"

MOST_FREQ_DIAGS_NUMS_OLD = [179,
                            198,
                            111,
                            127,
                            140,
                            8,
                            138,
                            185,
                            206,
                            186,
                            195,
                            207,
                            66,
                            157,
                            85]
MOST_FREQ_DIAGS_NUMS_NEW = [161,
                            158,
                            2,
                            156,
                            15,
                            44,
                            45,
                            157,
                            159,
                            60,
                            47,
                            46,
                            119,
                            123,
                            0]
MOST_FREQ_DIAGS_NAMES = ['non_specific_repolarisation_abnormalities_apical',
                         'non_specific_repolarisation_abnormalities_septal',
                         'sinus_bradycardia',
                         'non_specific_repolarisation_abnormalities_anterior_wall',
                         'atrial_fibrillation',
                         'electric_axis_vertical',
                         'electric_axis_horizontal',
                         'non_specific_repolarisation_abnormalities_lateral_wall',
                         'non_specific_repolarisation_abnormalities_inferior_wall',
                         'incomplete_right_bundle_branch_block',
                         'electric_axis_left_deviation',
                         'electric_axis_normal',
                         'right_atrial_hypertrophy',
                         'left_ventricular_hypertrophy',
                         'regular_normosystole']


def get_diag_dict():
    def deep(data, diag_list):
        for diag in data:
            if diag['type'] == 'diagnosis':
                diag_list.append(diag['name'])
            else:
                deep(diag['value'], diag_list)

    try:
        infile = open(DIAGNOSIS_DICTIONARY, 'rb')
        data = json.load(infile)

        diag_list = []
        deep(data, diag_list)

        diag_num = list(range(len(diag_list)))
        diag_dict = dict(zip(diag_list, diag_num))

        return diag_dict

    except FileNotFoundError:
        print("File " + DIAGNOSIS_DICTIONARY + " has not found.")
        sys.exit(0)


def load_dataset(is_old=True):
    if is_old:
        infile = open(OLD_PKL_PATH, 'rb')
        dataset = pkl.load(infile)
        infile.close()
    else:
        infile = open(OLD_AND_NEW_PKL_PATH, 'rb')
        (_, dataset) = pkl.load(infile)
        infile.close()

    return dataset


def normalize_data(x):
    mn = x.mean(axis=0)
    st = x.std(axis=0)
    x_std = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_std[i] = (x[i] - mn) / st
    return x_std


def load_dataset_6002(type='old'):
    if type == 'old':
        infile = open(OLD_6002_PATH, 'rb')
        xy = pkl.load(infile)
        infile.close()
    elif type == 'new':
        infile = open(NEW_6002_PATH, 'rb')
        xy = pkl.load(infile)
        infile.close()

    return xy
