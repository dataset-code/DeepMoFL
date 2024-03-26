import numpy as np
import keras
from keras.models import load_model, Model
import pandas as pd
import math
import time
import os
import sys
import json
import argparse

from compare import compare_classification, compare_regression

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

def calculate_confidence(output, is_classification, expected):
    if is_classification:
        if not np.isscalar(output) and len(output) == 1 and np.isnan(output):
            return 0
        if np.isscalar(output):
            return output         
        else:
            if np.isnan(np.max(output)):
                return 0
            else:
                return np.max(output) 
    else:
        if not np.isscalar(expected):
            expected = expected.flatten()
        if not np.isscalar(output):
            output = output.flatten()
        if np.isscalar(expected) and np.isscalar(output):
            return np.sqrt((expected - output)**2)
        if np.isscalar(expected):
            if len(output) > 1:
                return 0
            return np.sqrt((expected - output[0])**2)
        elif np.isscalar(output):
            if len(expected) > 1:
                return 0
            return np.sqrt((expected[0] - output)**2)
        if len(expected) != len(output):
            return 0
        res = 0
        for i in range(len(expected)):
            res = res + abs(expected[i] - output[i])**2
        res = np.sqrt(res)
        return res

def calculate_neuron_counts(model):
    neuron_counts = []
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            if isinstance(layer.output_shape, list):
                neuron_counts.append(int(np.prod(layer.output_shape[0][1:])))
            else:
                neuron_counts.append(int(np.prod(layer.output_shape[1:])))
    return neuron_counts

def select_neuron(neuron_counts, size = 500, seed = 7):
    np.random.seed(seed)
    selected_neuron = []
    new_neuron_counts = []
    for i in neuron_counts:
        if i > size:
            res = sorted(np.random.randint(low=0, high=i, size=size))
            selected_neuron.append(res)
            new_neuron_counts.append(size)
        else:
            selected_neuron.append([])
            new_neuron_counts.append(i)
    return selected_neuron, new_neuron_counts


formula_list = ['tarantula', 'ochiai', 'D_star', 'Op2', 'Barinel']

def main(x_test, y_test, model, classification, activation_threshold, res, size = 500, seed = 7, weighted=True):
    # get neuron number
    neuron_counts=calculate_neuron_counts(model)
    print("neuron_counts", neuron_counts, sum(neuron_counts))
    res['neuron_counts'] = neuron_counts

    selected_neuron, neuron_counts = select_neuron(neuron_counts, size, seed)
    print("selected_neuron", neuron_counts, sum(neuron_counts))
    res['neuron_counts_selected'] = neuron_counts

    # run test cases
    if isinstance(model.layers[0].input_shape, list):
        input_sample = x_test.reshape((len(x_test),)+model.layers[0].input_shape[0][1:])
    else:
        input_sample = x_test.reshape((len(x_test),)+model.layers[0].input_shape[1:])

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    if activation_model.layers[0].input_shape[0][0]:
        batch_num = math.ceil(len(x_test)/activation_model.layers[0].input_shape[0][0])
        activations = activation_model.predict(input_sample,steps=batch_num)
    else:
        activations = activation_model.predict(input_sample)
    
    # calculate activation matrix
    coverage_matrix = []   
    j = 0
    for layer_activation in activations:    # output of each layer (all test cases)
        new_layer_activation = []
        for i in range(len(x_test)):
            if len(selected_neuron[j]) == 0:
                new_layer_activation.append(layer_activation[i].flatten())
            else:
                new_layer_activation.append(layer_activation[i].flatten()[selected_neuron[j]])
        df = pd.DataFrame(new_layer_activation)
        if isinstance(coverage_matrix,list):
            coverage_matrix = df 
        else:
            coverage_matrix = pd.concat([coverage_matrix, df],axis = 1,ignore_index=True)   # concat row: test case, col: neuron
        j+=1

    coverage_matrix = coverage_matrix.applymap(lambda x: 1 if x > activation_threshold else 0)
   
   
    # ---- test cases confidence ----
    eva = []    # whether test cases passed 
    confidence = []
    for i in range(len(x_test)):
        if classification:
            if compare_classification(y_test[i], activations[-1][i]):
                eva.append(1)
            else:
                eva.append(0)    
        else:
            if compare_regression(y_test[i], activations[-1][i], 0.001):
                eva.append(1)
            else:
                eva.append(0)      

        confidence.append(calculate_confidence(activations[-1][i], classification, y_test[i]))        

    coverage_matrix['label']=eva
    coverage_matrix = coverage_matrix.astype(float)

    min_val = np.min(confidence)
    max_val = np.max(confidence)
    if min_val == max_val:
        confidence = [0 for _ in confidence]
    else:
        confidence = (confidence - min_val) / (max_val - min_val)
    if not classification:
        confidence = [1-x for x in confidence]

    if not weighted:
        confidence = [0 for _ in confidence]

    # ---- calculate neuron suspiciousness ----
    neuron_score_list = {}
    pass_fail = []

    total_pass = 1e-4 
    total_fail = 1e-4
    for i in range(len(eva)):
        if eva[i] == 1:
            total_pass+=(1+confidence[i])
        else:
            total_fail+=(1+confidence[i])

    for i in range(sum(neuron_counts)):
        passed=0
        failed=0
        for j in range(len(eva)):
            if coverage_matrix[i][j] == 1 and eva[j] == 1:
                passed+=(1+confidence[j])
            elif coverage_matrix[i][j] == 1 and eva[j] == 0:
                failed+=(1+confidence[j])
        pass_fail.append([passed, failed, total_pass, total_fail])    

        for i in formula_list:
            if i not in neuron_score_list:
                neuron_score_list[i] = [calculate_suspiciousness(passed, failed, total_pass, total_fail, i)]
            else:
                neuron_score_list[i].append(calculate_suspiciousness(passed, failed, total_pass, total_fail, i))
                
    # ---- calculate layer suspiciousness ----
    cur_count = 0

    layer_score_list = {}
    for i in neuron_counts:
        for j in formula_list:
            if j not in layer_score_list:
                layer_score_list[j] = [aggregate(neuron_score_list[j][cur_count:cur_count+i])]
            else:
                layer_score_list[j].append(aggregate(neuron_score_list[j][cur_count:cur_count+i]))
        cur_count+=i

    for i in formula_list:
        res[f'layer_{i}_score'] = layer_score_list[i]
        print(i, layer_score_list[i], len(layer_score_list[i]) - 1 - np.argmax(layer_score_list[i][::-1]))


def calculate_suspiciousness(passed, failed, total_pass, total_fail, formula):
    """
    passed(n_cs): number of successful test cases that cover a statement
    failed(n_cf): number of failed test cases that cover a statement
    total_pass(n_s): total number of successful test cases
    total_fail(n_f): total number of failed test cases
    """
    if passed == 0 and failed == 0:
        return 0
    else:
        if formula == 'tarantula':
            return (failed/total_fail)/((failed/total_fail)+(passed/total_pass))
        elif formula == 'ochiai':
            return failed/math.sqrt(total_fail*(passed+failed))
        elif formula == 'D_star':
            return failed*failed/(total_fail-failed+passed)
        elif formula == 'Op2':
            return failed-passed/(total_pass+1)
        elif formula == 'Barinel':
            return 1-passed/(passed+failed)

def aggregate(score_list):
    score = np.mean(score_list)*(1-gini_coefficient(np.array(score_list)))
    if np.isnan(score):
        score = 0
    return score   

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))




