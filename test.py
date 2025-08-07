import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np
# ---------------------- ฟังชั่นที่ใช้ ----------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# ---------------------- โหลดข้อมูล  ----------------------

def load_flood_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # ไม่อ่านหัวไฟล์
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


# ---------------------- Normalization  ข่้อมูล ----------------------

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

def initialize_network(input_size, hidden_sizes, output_size, seed=None): #กำหนดเงื่อนไขของ initialize weight
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

def forward_propagate(network, inputs): #ทำ forward 
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

def backward_propagate_error(network, expected): #ทำ backward
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

def update_weights(network, row, l_rate, momentum, prev_updates): #update weight หลังจบแต่ละรอบ
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

def train_network(network, dataset, l_rate, momentum, n_epoch, n_outputs, track_error=False): #train ข้อมูล
    prev_updates = [
        [[0.0 for _ in neuron['weights']] for neuron in layer] for layer in network
    ]
    error_history = []
    
    for epoch in range(n_epoch):
        total_error = 0
        for row in dataset:
            outputs = forward_propagate(network, row[:-1])
            expected = [row[-1]] if n_outputs == 1 else [0 for _ in range(n_outputs)]
            if n_outputs > 1:
                expected[int(row[-1])] = 1
            total_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
            backward_propagate_error(network, expected)
            update_weights(network, row[:-1], l_rate, momentum, prev_updates)
        avg_error = total_error / len(dataset)
        if track_error:
            error_history.append(avg_error)
    
    if track_error:
        return avg_error, error_history
    else:
        return avg_error


def predict(network, row): # การpredict ข้อมูล
    outputs = forward_propagate(network, row)
    return outputs[0] if len(outputs) == 1 else outputs.index(max(outputs))

# ---------------------- ทำ Cross Validation ----------------------

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

# ---------------------- ทำ Confusion Matrix ----------------------

def confusion_matrix(actual, predicted, num_classes=2):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(actual)):
        matrix[actual[i]][predicted[i]] += 1
    return matrix

def plot_confusion_matrix(confusion_matrix, experiment_name):
    """
    Plot confusion matrix using matplotlib
    """
    plt.figure(figsize=(8, 6))
    
    # Convert to numpy array for easier handling
    cm = np.array(confusion_matrix)
    
    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # Add labels
    classes = ['Class 0', 'Class 1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"confusion_matrix_{experiment_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as: {plot_filename}")
    plt.show()

#  ฟังก์ชันหลักในการทดสอบ Neural Network

def run_experiment_detailed(dataset, task='regression', hidden_sizes=[4], l_rate=0.1, momentum=0.9,
                           n_epoch=100, n_folds=10, min_vals=None, max_vals=None, weight_seed=None, 
                           experiment_name="", plot_training=False, plot_confusion=False):
    print(f"\n=== {experiment_name} ===")
    print(f"Hidden Layers: {hidden_sizes}, Learning Rate: {l_rate}, Momentum: {momentum}, Epochs: {n_epoch}, Seed: {weight_seed}")
    
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    confusion = [[0, 0], [0, 0]] if task == 'classification' else None
    
    fold_results = []
    all_error_histories = []  # เก็บ error history ของทุก fold
    
    for fold_i, fold in enumerate(folds, start=1):
        train_set = [row for f in folds if f != fold for row in f]
        test_set = [list(row) for row in fold]
        
        n_inputs = len(train_set[0]) - 1
        n_outputs = 1 if task == 'regression' else 2
        
        network = initialize_network(n_inputs, hidden_sizes, n_outputs, seed=weight_seed)
        
        # Train with error tracking for regression
        if task == 'regression' and plot_training:
            train_error, error_history = train_network(network, train_set, l_rate, momentum, n_epoch, n_outputs, track_error=True)
            all_error_histories.append(error_history)
        else:
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
            fold_results.append(f"Fold {fold_i:2d}: MSE = {mse:.6f}, Train Error = {train_error:.6f}")
        else:
            # สำหรับ Classification
            for i in range(len(predictions)):
                confusion[int(actuals[i])][int(predictions[i])] += 1
            correct = sum(1 for i in range(len(predictions)) if int(predictions[i]) == int(actuals[i]))
            acc = correct / len(predictions)
            scores.append(acc)
            fold_results.append(f"Fold {fold_i:2d}: Accuracy = {acc*100:.2f}%, Train Error = {train_error:.6f}")
    
    # Print fold results
    for result in fold_results:
        print(result)
    
    # Plot กราฟของ regression
    if task == 'regression' and plot_training and all_error_histories:
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(range(n_folds))
        
        for fold_i, (error_history, color) in enumerate(zip(all_error_histories, colors), 1):
            plt.plot(range(1, len(error_history) + 1), error_history, 
                    color=color, alpha=0.7, linewidth=1.5, label=f'Fold {fold_i}')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title(f'Training MSE vs Epoch - {experiment_name}', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # ใช้ log scale เพราะ MSE มักจะลดลงแบบ exponential
        plt.tight_layout()
        
        # Save รูปภาพของ plot
        plot_filename = f"training_curve_{experiment_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training curve saved as: {plot_filename}")
        plt.show()
        
        # print convergence
        final_errors = [history[-1] for history in all_error_histories]
        print(f"Final training MSE - Mean: {sum(final_errors)/len(final_errors):.6f}, "
              f"Std: {(sum((x - sum(final_errors)/len(final_errors))**2 for x in final_errors) / len(final_errors))**0.5:.6f}")
    
    # สรุปผลprintออก terminal
    if task == 'regression':
        avg_mse = sum(scores) / len(scores)
        std_mse = (sum((x - avg_mse)**2 for x in scores) / len(scores)) ** 0.5
        print(f"Average Test MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    else:
        avg_acc = sum(scores) / len(scores)
        std_acc = (sum((x - avg_acc)**2 for x in scores) / len(scores)) ** 0.5
        print(f"Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print("Overall Confusion Matrix:")
        print(f"           Predicted")
        print(f"         Class0  Class1")
        print(f"Actual Class0: {confusion[0][0]:4d}   {confusion[0][1]:4d}")
        print(f"       Class1: {confusion[1][0]:4d}   {confusion[1][1]:4d}")
        
        # Plot confusion matrix if requested
        if plot_confusion:
            plot_confusion_matrix(confusion, experiment_name)
        
        # คำนวน metrics
        tp, tn, fp, fn = confusion[1][1], confusion[0][0], confusion[0][1], confusion[1][0]
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    print("-" * 80)
    return scores

#ตัวหลักสำหรับไปเรียกรันการทดลองทั้งหมด
def comprehensive_experiments():
    print("=" * 80)
    
    # โหลดข้อมูล
    print("Loading datasets...")
    flood_data = load_flood_data('Flood_data.csv')
    flood_data, flood_min, flood_max = normalize_dataset(flood_data)
    
    cross_data = load_cross_pat('cross.pat')
    
    # ============ Flood data set ============
    print("\n" + "="*50)
    print("Flood data set ")
    print("="*50)
    
    # 1. Hidden Layer 
    hidden_configs = [
        [4], [5], [6],     # 1 layer
        [4, 3], [5, 3],    # 2 layers  
        [8, 4, 2],         # 3 layers
    ]
    
    # เปลี่ยนจำนวน Hidden Nodes
    for config in hidden_configs:
        run_experiment_detailed(flood_data, task='regression', hidden_sizes=config, 
                              l_rate=0.05, momentum=0.8, n_epoch=1000, n_folds=10,
                              min_vals=flood_min, max_vals=flood_max, weight_seed=42,
                              experiment_name=f"Hidden Nodes {config}", 
                              plot_training=True)  
    
    # เปลี่ยน Learning Rate
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    for lr in learning_rates:
        run_experiment_detailed(flood_data, task='regression', hidden_sizes=[4], 
                              l_rate=lr, momentum=0.8, n_epoch=1000, n_folds=10,
                              min_vals=flood_min, max_vals=flood_max, weight_seed=42,
                              experiment_name=f"Learning Rate {lr}",
                              plot_training=True)  
    
    # เปลี่ยน Momentum 
    momentum_values = [0.0, 0.3, 0.5, 0.7,]
    for mom in momentum_values:
        run_experiment_detailed(flood_data, task='regression', hidden_sizes=[4], 
                              l_rate=0.05, momentum=mom, n_epoch=1000, n_folds=10,
                              min_vals=flood_min, max_vals=flood_max, weight_seed=42,
                              experiment_name=f"Momentum {mom}",plot_training=True)
    
    # เปลี่ยน Weight 
    weight_seeds = [1, 7, 42, 99, 123]
    for seed in weight_seeds:
        run_experiment_detailed(flood_data, task='regression', hidden_sizes=[4], 
                              l_rate=0.05, momentum=0.8, n_epoch=1000, n_folds=10,
                              min_vals=flood_min, max_vals=flood_max, weight_seed=seed,
                              experiment_name=f"Weight Seed {seed}",plot_training=True)
    
    # ============ การทดลอง cross.pat============
    print("\n" + "="*50)
    print("CROSS.PAT ")
    print("="*50)
    
    # เปลี่ยนจำนวน Hidden Nodes
    for config in hidden_configs[:6]:  
        run_experiment_detailed(cross_data, task='classification', hidden_sizes=config, 
                              l_rate=0.1, momentum=0.9, n_epoch=1000, n_folds=10, 
                              weight_seed=1, experiment_name=f"Hidden Nodes {config}",
                              plot_confusion=True)  # เพิ่ม plot_confusion=True
    
    # เปลี่ยน Learning Rate  
    for lr in learning_rates:
        run_experiment_detailed(cross_data, task='classification', hidden_sizes=[4], 
                              l_rate=lr, momentum=0.9, n_epoch=1000, n_folds=10, 
                              weight_seed=1, experiment_name=f"Learning Rate {lr}",
                              plot_confusion=True)  # เพิ่ม plot_confusion=True
    
    # เปลี่ยน Momentum 
    for mom in momentum_values:
        run_experiment_detailed(cross_data, task='classification', hidden_sizes=[4], 
                              l_rate=0.1, momentum=mom, n_epoch=1000, n_folds=10, 
                              weight_seed=1, experiment_name=f"Momentum {mom}",
                              plot_confusion=True)  # เพิ่ม plot_confusion=True
    
    # เปลี่ยน Weight 
    for seed in weight_seeds:
        run_experiment_detailed(cross_data, task='classification', hidden_sizes=[4], 
                              l_rate=0.1, momentum=0.9, n_epoch=1000, n_folds=10, 
                              weight_seed=seed, experiment_name=f"Weight Seed {seed}",
                              plot_confusion=True)  # เพิ่ม plot_confusion=True

# function main 
if __name__ == '__main__':
    comprehensive_experiments()