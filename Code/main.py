import numpy as np
import keras
from keras.models import load_model, Model
import pandas as pd
import math
import time
import os
import sys
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

def compare_classification(expected, actual):
    # print(expected, actual )
    if not np.isscalar(actual) and len(actual) == 1 and np.isnan(actual):
        return False
    if not np.isscalar(expected) and not np.isscalar(actual) and len(expected) == 1 and len(actual) == 1:
        expected = expected[0]
        actual == actual[0]
    if len(actual.flatten()) == 1:
        actual = actual.flatten()[0]
    if len(expected.flatten()) == 1:
        expected = expected.flatten()[0]

    if np.isscalar(expected) and np.isscalar(actual):
        return expected == round(actual)
    if np.isscalar(expected):
        if len(actual) == 1:
            return expected == round(actual[0])    
        return expected == np.argmax(actual)
    elif np.isscalar(actual):
        if len(expected) == 1:
            return expected[0] == round(actual)
        return np.argmax(expected) == round(actual)
    if len(expected) != len(actual):
        print("[????]")
        return False
    return np.argmax(expected) == np.argmax(actual)

def compare_regression(expected, actual, delta):
    if not np.isscalar(expected):
        expected = expected.flatten()
    if not np.isscalar(actual):
        actual = actual.flatten()

    if np.isscalar(expected) and np.isscalar(actual):
        return abs(expected - actual) <= delta
    if np.isscalar(expected):
        if len(actual) > 1:
            return False
        return abs(expected - actual[0]) <= delta
    elif np.isscalar(actual):
        if len(expected) > 1:
            return False
        return abs(expected[0] - actual) <= delta
    n = len(expected)
    if n != len(actual):
        return False
    res = True
    for i in range(0, n):
        res = res and abs(expected[i] - actual[i]) <= delta
    return res

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
    # cur_pos = 0
    for i in neuron_counts:
        if i > size:
            res = sorted(np.random.randint(low=0, high=i, size=size))
            # res = [j+cur_pos for j in res]
            # selected_neuron.extend(res)
            selected_neuron.append(res)
            new_neuron_counts.append(size)
        else:
            # res = [j+cur_pos for j in range(i)]
            # selected_neuron.extend(res)
            selected_neuron.append([])
            new_neuron_counts.append(i)
        # cur_pos+=i
    return selected_neuron, new_neuron_counts


formula_list = ['tarantula', 'ochiai', 'D_star', 'Op2', 'Barinel']

def main(id, x_test, y_test, model, classification, activation_threshold, res, size = 500, seed = 7, weighted=True):
    # 计算神经元数量
    neuron_counts=calculate_neuron_counts(model)
    print("neuron_counts", neuron_counts, sum(neuron_counts))
    res['neuron_counts'] = neuron_counts

    selected_neuron, neuron_counts = select_neuron(neuron_counts, size, seed)
    print("selected_neuron", neuron_counts, sum(neuron_counts))
    res['neuron_counts_selected'] = neuron_counts

    # 执行测试用例
    if isinstance(model.layers[0].input_shape, list):
        input_sample = x_test.reshape((len(x_test),)+model.layers[0].input_shape[0][1:])
    else:
        input_sample = x_test.reshape((len(x_test),)+model.layers[0].input_shape[1:])

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # print(activation_model.summary())
    if activation_model.layers[0].input_shape[0][0]:
        batch_num = math.ceil(len(x_test)/activation_model.layers[0].input_shape[0][0])
        activations = activation_model.predict(input_sample,steps=batch_num)
    else:
        activations = activation_model.predict(input_sample)
    
    # 计算激活矩阵
    coverage_matrix = []    # 每个测试用例的神经元激活情况
    j = 0
    for layer_activation in activations:    # 每层的输出（所有测试用例）
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
            coverage_matrix = pd.concat([coverage_matrix, df],axis = 1,ignore_index=True)   # 拼接输出矩阵，每一行为一个测试用例，每一列为一个神经元
        j+=1

    coverage_matrix = coverage_matrix.applymap(lambda x: 1 if x > activation_threshold else 0)   # 判断神经元是否激活
   
   
    # 判断测试用例是否通过
    eva = []    # 测试用例是否通过
    confidence = []
    for i in range(len(x_test)):
        # 取最后一层的输出，跟标签比较，并计算confidence
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
    # print("【eva】", eva)
    # print(confidence)
    min_val = np.min(confidence)
    max_val = np.max(confidence)
    if min_val == max_val:
        confidence = [0 for _ in confidence]
    else:
        confidence = (confidence - min_val) / (max_val - min_val)
    if not classification:
        confidence = [1-x for x in confidence]
    # confidence = [min(x+1e-8, 1) for x in confidence]
    # print("【confidence】", confidence)
    # print(activations[-1])
    # print(coverage_matrix)
    if not weighted:
        confidence = [0 for _ in confidence]

    # ---- 计算每个神经元的可疑值 ----
    neuron_score_list = {}
    pass_fail = []

    total_pass = 1e-4 # 避免除0错误
    total_fail = 1e-4
    for i in range(len(eva)):
        if eva[i] == 1:
            total_pass+=(1+confidence[i])
        else:
            total_fail+=(1+confidence[i])
    print(f"total_pass and fail")

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

    # print("【neuron suspiousness】",neuron_score_list)
    # ---- 计算每一层的可疑值 ----
    # 取每个神经元的平均值作为每一层的可疑值
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
    return score   

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def run(id, is_classification, selected_neuron_num, activation_threshold, seed):

    x_test = np.load(f"../Dataset/{id}/x_test.npy")
    y_test = np.load(f"../Dataset/{id}/y_test.npy")
    model = keras.models.load_model(f"../Dataset/{id}/model.h5")

    res = {}
    t1 = time.time()
    main(id, x_test, y_test, model, is_classification, activation_threshold, res, selected_neuron_num, seed)
    t2 = time.time()
    res['time'] = t2-t1
    print(f"time: {t2-t1}")

    # print(res)
    with open(f'res_{activation_threshold}_{selected_neuron_num}_{seed}_gini.json',"w",encoding='utf-8') as f:
        json.dump(res,f)

if __name__ == "__main__":
    id = sys.argv[1]
    is_classification = sys.argv[2]
    selected_neuron_num = int(sys.argv[3])
    activation_threshold = float(sys.argv[4])
    seed = int(sys.argv[5])
    run(id, is_classification, selected_neuron_num, activation_threshold, seed)



