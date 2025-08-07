import csv
import random
import math

# ---------------------- Utils ----------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# ---------------------- Data Loaders ----------------------

def load_flood_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # ข้าม header
        for row in reader:
            if row:
                dataset.append([float(x) for x in row])
    return dataset

def load_cross_pat(filename):
    dataset = []
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
        for i in range(0, len(lines), 3):
            inputs = [float(x) for x in lines[i+1].split()]
            outputs = [int(x) for x in lines[i+2].split()]
            class_label = outputs.index(1)
            dataset.append(inputs + [class_label])
    return dataset


# ---------------------- Normalization ----------------------

def normalize_dataset(dataset):
    min_vals = [min(col) for col in zip(*dataset)]
    max_vals = [max(col) for col in zip(*dataset)]

    normalized = []
    for row in dataset:
        norm_row = []
        for i in range(len(row)):
            if max_vals[i] == min_vals[i]:
                norm_row.append(0.0)
            else:
                norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
        normalized.append(norm_row)
    return normalized, min_vals, max_vals

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# ---------------------- Neural Network ----------------------

def initialize_network(input_size, hidden_sizes, output_size, seed=None):
    if seed is not None:
        random.seed(seed)  # กำหนด seed เพื่อให้สุ่มเหมือนเดิมทุกครั้ง
    network = []
    prev_size = input_size
    for hidden_size in hidden_sizes:
        layer = [{'weights': [random.uniform(-1, 1) for _ in range(prev_size + 1)]} for _ in range(hidden_size)]
        network.append(layer)
        prev_size = hidden_size
    output_layer = [{'weights': [random.uniform(-1, 1) for _ in range(prev_size + 1)]} for _ in range(output_size)]
    network.append(output_layer)
    return network

def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron['weights'][-1]
            for i in range(len(neuron['weights']) - 1):
                activation += neuron['weights'][i] * inputs[i]
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network) - 1:
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        else:
            for j in range(len(layer)):
                error = sum(n['weights'][j] * n['delta'] for n in network[i + 1])
                errors.append(error)
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

def update_weights(network, row, l_rate, momentum, prev_updates):
    inputs = row
    for i, layer in enumerate(network):
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for j, neuron in enumerate(layer):
            for k in range(len(inputs)):
                delta = l_rate * neuron['delta'] * inputs[k]
                neuron['weights'][k] += delta + momentum * prev_updates[i][j][k]
                prev_updates[i][j][k] = delta
            delta = l_rate * neuron['delta']
            neuron['weights'][-1] += delta + momentum * prev_updates[i][j][-1]
            prev_updates[i][j][-1] = delta

def train_network(network, dataset, l_rate, momentum, n_epoch, n_outputs):
    prev_updates = [
        [[0.0 for _ in neuron['weights']] for neuron in layer] for layer in network
    ]
    total_error = 0
    for epoch in range(n_epoch):
        total_error = 0
        for row in dataset:
            outputs = forward_propagate(network, row[:-1])
            expected = [row[-1]] if n_outputs == 1 else [0 for _ in range(n_outputs)]
            if n_outputs > 1:
                expected[int(row[-1])] = 1
            total_error += sum((expected[i] - outputs[i])**2 for i in range(len(expected)))
            backward_propagate_error(network, expected)
            update_weights(network, row[:-1], l_rate, momentum, prev_updates)
    return total_error / len(dataset)  # คืนค่า MSE เฉลี่ย

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs[0] if len(outputs) == 1 else outputs.index(max(outputs))

# ---------------------- Cross Validation ----------------------

def cross_validation_split(dataset, n_folds=10):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size and dataset_copy:
            i = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(i))
        dataset_split.append(fold)
    return dataset_split

# ---------------------- Confusion Matrix ----------------------

def confusion_matrix(actual, predicted, num_classes=2):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(actual)):
        matrix[actual[i]][predicted[i]] += 1
    return matrix

# ---------------------- Main Experiment ----------------------

def run_experiment(dataset, task='regression', hidden_sizes=[4], l_rate=0.1, momentum=0.9,
                   n_epoch=100, n_folds=10, min_vals=None, max_vals=None, weight_seed=None):
    print(f"Running experiment with hidden nodes: {hidden_sizes} | weight_seed = {weight_seed}")
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    confusion = [[0, 0], [0, 0]]

    for fold_i, fold in enumerate(folds, start=1):
        train_set = [row for f in folds if f != fold for row in f]
        test_set = [list(row) for row in fold]

        n_inputs = len(train_set[0]) - 1
        n_outputs = 1 if task == 'regression' else 2

        # ใช้ seed ถ้ามี
        network = initialize_network(n_inputs, hidden_sizes, n_outputs, seed=weight_seed)

        train_error = train_network(network, train_set, l_rate, momentum, n_epoch, n_outputs)

        predictions = []
        actuals = []

        for row in test_set:
            prediction = predict(network, row[:-1])
            actual_val = row[-1] if task == 'regression' else int(row[-1])
            predictions.append(prediction)
            actuals.append(actual_val)

        if task == 'regression':
            mse = sum((actuals[i] - predictions[i])**2 for i in range(len(actuals))) / len(actuals)
            scores.append(mse)
            print(f"Fold {fold_i} Test MSE: {mse:.4f}") 
        else:
            for i in range(len(predictions)):
                confusion[int(actuals[i])][int(predictions[i])] += 1
            correct = sum(1 for i in range(len(predictions)) if int(predictions[i]) == int(actuals[i]))
            acc = correct / len(predictions)
            scores.append(acc)
            print(f"Fold {fold_i} Accuracy: {acc*100:.2f}%")

    if task == 'regression':
        print(f"Average Test MSE: {sum(scores)/len(scores):.4f}")
    else:
        print(f"Confusion Matrix (hidden nodes = {hidden_sizes}):")
        for row in confusion:
            print(row)
        print(f"Average Accuracy: {sum(scores)/len(scores)*100:.2f}%")


# ---------------------- RUN ----------------------

if __name__ == '__main__':
    # -------- Flood_data.csv --------
    print("1. Flood_data.csv (Regression)")
    flood_data = load_flood_data('Flood_data.csv')
    flood_data, flood_min, flood_max = normalize_dataset(flood_data)


            # การเปลี่ยนแปลง Hidden nodes  โดย learning rate,momentum,n_epoch,n_folds,min_vals,max_vals เท่าเดิม แต่ weight_seed เปลี่ยน
    print("------------------------------- เปลี่ยนจำนวน Hidden Nodes -------------------------------")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.01, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=None)
    run_experiment(flood_data, task='regression', hidden_sizes=[5], l_rate=0.01, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=None)
    run_experiment(flood_data, task='regression', hidden_sizes=[4,3], l_rate=0.01, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=None)
    
            # การเปลี่ยนแปลง learning rate โดย momentum,n_epoch,n_folds,min_vals,max_vals เท่าเดิม กำหนด weight_seed=42เพิ่มไม่ให้randomใหม่
    print("------------------------------- เปลี่ยน Learning Rate -------------------------------")
    print("learning rate=0.01")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.01, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)
    print("learning rate=0.05")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.05, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)
    print("learning rate=0.10")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.10, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)

            # การเปลี่ยนแปลง Momentum rate โดย n_epoch,n_folds,min_vals,max_vals เท่าเดิม กำหนด weight_seed=42เพิ่มไม่ให้randomใหม่
    print("------------------------------- เปลี่ยน Momentum -------------------------------")
    print("momentum=0.6")
    run_experiment(flood_data, task='regression', hidden_sizes= [4], l_rate=0.05, momentum=0.6, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)
    print("momentum=0.7")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.05, momentum=0.7, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)
    print("momentum=0.8")
    run_experiment(flood_data, task='regression', hidden_sizes=[4], l_rate=0.05, momentum=0.8, n_epoch=1000, n_folds=10,
                    min_vals=flood_min, max_vals=flood_max,weight_seed=42)



    # -------- ทดลองกับ cross.pat --------
    print("\n2. ทดลองกับ cross.pat")
    cross_data = load_cross_pat('cross.pat')

    print("\n------------------------------- ทดลอง Base Line -------------------------------")
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.1, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=1)

    print("\n------------------------------- เปลี่ยนจำนวน Hidden Nodes -------------------------------")
    run_experiment(cross_data, task='classification', hidden_sizes=[3], l_rate=0.1, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=1)
    run_experiment(cross_data, task='classification', hidden_sizes=[4, 2], l_rate=0.1, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=1)

    print("\n------------------------------- เปลี่ยน Learning Rate -------------------------------")
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.01, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=1)
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.5, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=1)

    print("\n------------------------------- เปลี่ยน Momentum -------------------------------")
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.1, momentum=0.6,
                   n_epoch=1000, n_folds=10, weight_seed=1)
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.1, momentum=0.95,
                   n_epoch=1000, n_folds=10, weight_seed=1)

    print("\n------------------------------- เปลี่ยนค่า Initial Weights -------------------------------")
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.1, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=7)
    run_experiment(cross_data, task='classification', hidden_sizes=[4], l_rate=0.1, momentum=0.9,
                   n_epoch=1000, n_folds=10, weight_seed=99)