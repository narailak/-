# ===================== ใส่ชื่อ Dataset  =====================
DATASET_CONFIG = {
    'flood_data_file': 'Flood_data.csv',
    'cross_data_file': 'cross.pat'
}

# ===================== Config Hidden nodes =====================
HIDDEN_CONFIGS = [
    [4], [5], [6],     # 1 layer
    [4, 3], [5, 3],    # 2 layers  
    [8, 4, 2],[4, 4, 2]        # 3 layers
]

# =====================  Config Parameters =====================
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]
MOMENTUM_VALUES = [0.0, 0.3, 0.5, 0.7]
WEIGHT_SEEDS = [1, 7, 42, 99, 123]

# ===================== Training Config =====================
TRAINING_CONFIG = {
    'n_epochs': 1000,
    'n_folds': 10,
}

# ===================== Default Parameters =====================
DEFAULT_PARAMS = {
    'flood': {
        'hidden_sizes': [4],
        'l_rate': 0.05,
        'momentum': 0.8,
        'weight_seed': 42,
        'plot_training': True,
        'plot_confusion': False
    },
    'cross': {
        'hidden_sizes': [4],
        'l_rate': 0.1,
        'momentum': 0.9,
        'weight_seed': 1,
        'plot_training': False,
        'plot_confusion': True
    }
}

# ===================== config การทดลองแต่ละประเภท =====================
FLOOD_EXPERIMENTS = [
    {
        'name': 'Hidden Architecture',
        'description': 'Testing different hidden layer architectures',
        'params': {
            'variable': 'hidden_sizes',
            'values': HIDDEN_CONFIGS,
            'fixed_params': {
                'l_rate': DEFAULT_PARAMS['flood']['l_rate'],
                'momentum': DEFAULT_PARAMS['flood']['momentum'],
                'weight_seed': DEFAULT_PARAMS['flood']['weight_seed'],
                'plot_training': True
            }
        }
    },
    {
        'name': 'Learning Rate',
        'description': 'Testing different learning rates',
        'params': {
            'variable': 'l_rate',
            'values': LEARNING_RATES,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['flood']['hidden_sizes'],
                'momentum': DEFAULT_PARAMS['flood']['momentum'],
                'weight_seed': DEFAULT_PARAMS['flood']['weight_seed'],
                'plot_training': True
            }
        }
    },
    {
        'name': 'Momentum',
        'description': 'Testing different momentum values',
        'params': {
            'variable': 'momentum',
            'values': MOMENTUM_VALUES,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['flood']['hidden_sizes'],
                'l_rate': DEFAULT_PARAMS['flood']['l_rate'],
                'weight_seed': DEFAULT_PARAMS['flood']['weight_seed'],
                'plot_training': True
            }
        }
    },
    {
        'name': 'Weight Initialization',
        'description': 'Testing different weight initialization seeds',
        'params': {
            'variable': 'weight_seed',
            'values': WEIGHT_SEEDS,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['flood']['hidden_sizes'],
                'l_rate': DEFAULT_PARAMS['flood']['l_rate'],
                'momentum': DEFAULT_PARAMS['flood']['momentum'],
                'plot_training': True
            }
        }
    }
]

CROSS_EXPERIMENTS = [
    {
        'name': 'Hidden Architecture',
        'description': 'Testing different hidden layer architectures',
        'params': {
            'variable': 'hidden_sizes',
            'values': HIDDEN_CONFIGS[:6],  # ใช้แค่ 6 config แรก
            'fixed_params': {
                'l_rate': DEFAULT_PARAMS['cross']['l_rate'],
                'momentum': DEFAULT_PARAMS['cross']['momentum'],
                'weight_seed': DEFAULT_PARAMS['cross']['weight_seed'],
                'plot_confusion': True
            }
        }
    },
    {
        'name': 'Learning Rate',
        'description': 'Testing different learning rates',
        'params': {
            'variable': 'l_rate',
            'values': LEARNING_RATES,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['cross']['hidden_sizes'],
                'momentum': DEFAULT_PARAMS['cross']['momentum'],
                'weight_seed': DEFAULT_PARAMS['cross']['weight_seed'],
                'plot_confusion': True
            }
        }
    },
    {
        'name': 'Momentum',
        'description': 'Testing different momentum values',
        'params': {
            'variable': 'momentum',
            'values': MOMENTUM_VALUES,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['cross']['hidden_sizes'],
                'l_rate': DEFAULT_PARAMS['cross']['l_rate'],
                'weight_seed': DEFAULT_PARAMS['cross']['weight_seed'],
                'plot_confusion': True
            }
        }
    },
    {
        'name': 'Weight Initialization',
        'description': 'Testing different weight initialization seeds',
        'params': {
            'variable': 'weight_seed',
            'values': WEIGHT_SEEDS,
            'fixed_params': {
                'hidden_sizes': DEFAULT_PARAMS['cross']['hidden_sizes'],
                'l_rate': DEFAULT_PARAMS['cross']['l_rate'],
                'momentum': DEFAULT_PARAMS['cross']['momentum'],
                'plot_confusion': True
            }
        }
    }
]

# print ค่าการทดลอง
def get_experiment_name(experiment_type, variable, value):
    if variable == 'hidden_sizes':
        return f"Hidden Nodes {value}"
    elif variable == 'l_rate':
        return f"Learning Rate {value}"
    elif variable == 'momentum':
        return f"Momentum {value}"
    elif variable == 'weight_seed':
        return f"Weight Seed {value}"
    else:
        return f"{variable} {value}"

# print ประเภทของการทดลอง
def print_experiment_header(dataset_name):
    print("\n" + "="*50)
    print(f"{dataset_name}")
    print("="*50)

#แสดงตอนเริ่มโปรแกรม Print การตั้งค่าทั้งหมด
def print_config_summary():
    print("="*80)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Hidden Architectures: {len(HIDDEN_CONFIGS)} configs")
    for i, config in enumerate(HIDDEN_CONFIGS, 1):
        print(f"  {i}. {config}")
    print(f"\nLearning Rates: {LEARNING_RATES}")
    print(f"Momentum Values: {MOMENTUM_VALUES}")
    print(f"Weight Seeds: {WEIGHT_SEEDS}")
    print(f"Training Epochs: {TRAINING_CONFIG['n_epochs']}")
    print(f"Cross-Validation Folds: {TRAINING_CONFIG['n_folds']}")
    print("="*80)