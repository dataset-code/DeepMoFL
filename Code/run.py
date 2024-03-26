from main import main
import numpy as np
import keras
import time
import json
import argparse

def run(model, x_test, y_test, is_classification, selected_neuron_num, activation_threshold, seed):
    x_test = np.load(x_test)
    y_test = np.load(y_test)
    model = keras.models.load_model(model)

    res = {}
    t1 = time.time()
    main(x_test, y_test, model, is_classification, activation_threshold, res, selected_neuron_num, seed)
    t2 = time.time()
    res['time'] = t2-t1
    print(f"time: {t2-t1}")

    with open(f'res_{activation_threshold}_{selected_neuron_num}_{seed}_gini.json',"w",encoding='utf-8') as f:
        json.dump(res,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required= True, help='path to model.h5')
    parser.add_argument('-i', '--input', type=str, required= True, help='path to inputs.npy')
    parser.add_argument('-o', '--output', type=str, required= True, help='path to output.npy')
    parser.add_argument('-c', '--classification', type=str, required= True, choices=['0','1'],help='whether the model is a classification model')
    parser.add_argument('-n', '--selected_neuron_num', type=int, required= True, help='selected neuron num')
    parser.add_argument('-t', '--threshold', type=float, required= True, help='activation threshold')
    parser.add_argument('-s', '--seed', type=int, required= True, help='seed for randomly selecting neuron')

    args = parser.parse_args()

    run(args.model, args.input, args.output, args.classification, args.selected_neuron_num, args.threshold, args.seed)