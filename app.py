"""
Flask Web Application for PSO-ANN
Interactive interface to run experiments and visualize results
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from pathlib import Path
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ann import NeuralNetwork
from pso import PSO
from pso_ann_trainer import rmse, create_fitness_function, to_numpy

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load data on startup
data_dir = Path(__file__).resolve().parent / 'data'
npz_path = data_dir / 'processed_data.npz'

if npz_path.exists():
    data = np.load(npz_path, allow_pickle=True)
    X_train_global = to_numpy(data['X_train'])
    X_test_global = to_numpy(data['X_test'])
    y_train_global = to_numpy(data['y_train']).ravel()
    y_test_global = to_numpy(data['y_test']).ravel()
    DATA_LOADED = True
else:
    DATA_LOADED = False


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/data-info')
def data_info():
    """Get information about the loaded dataset."""
    if not DATA_LOADED:
        return jsonify({'error': 'Data not loaded'}), 500
    
    return jsonify({
        'train_samples': int(X_train_global.shape[0]),
        'test_samples': int(X_test_global.shape[0]),
        'features': int(X_train_global.shape[1]),
        'target_min': float(y_train_global.min()),
        'target_max': float(y_train_global.max()),
        'target_mean': float(y_train_global.mean()),
        'target_std': float(y_train_global.std())
    })


@app.route('/api/run-experiment', methods=['POST'])
def run_experiment():
    """Run PSO-ANN optimization with user-provided parameters."""
    if not DATA_LOADED:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        params = request.json
        
        # Parse parameters
        layer_sizes = params.get('layer_sizes', [8, 64, 32, 1])
        activation_functions = params.get('activation_functions', ['relu', 'relu', 'linear'])
        swarm_size = int(params.get('swarm_size', 30))
        num_informants = int(params.get('num_informants', 3))
        max_iterations = int(params.get('max_iterations', 50))
        bounds = (-1.0, 1.0)  # Fixed bounds
        
        # Validate activation functions
        valid_activations = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'linear']
        for act in activation_functions:
            if act not in valid_activations:
                return jsonify({'error': f'Invalid activation function: {act}. Valid options: {", ".join(valid_activations)}'}), 400
        
        # Validate layer_sizes matches number of activation functions
        if len(layer_sizes) - 1 != len(activation_functions):
            return jsonify({'error': 'Number of activation functions must be (number of layers - 1)'}), 400
        
        # Create network
        network = NeuralNetwork(layer_sizes, activation_functions)
        network_info = network.get_network_info()
        
        # Calculate initial RMSE
        y_pred_initial = network.predict(X_train_global)
        if y_pred_initial.ndim > 1:
            y_pred_initial = y_pred_initial.ravel()
        rmse_initial = float(rmse(y_train_global, y_pred_initial))
        
        # Create fitness function
        fitness_func = create_fitness_function(network, X_train_global, y_train_global)
        
        # Run PSO
        pso = PSO(
            objective_function=fitness_func,
            dimension=network_info['total_parameters'],
            swarm_size=swarm_size,
            num_informants=num_informants,
            w=0.729,
            c1=1.49445,
            c2=1.49445,
            bounds=bounds,
            max_iterations=max_iterations
        )
        
        best_params, best_fitness, fitness_history = pso.optimize(verbose=False)
        
        # Apply best parameters
        network.set_parameters(best_params)
        
        # Evaluate
        y_pred_train = network.predict(X_train_global)
        if y_pred_train.ndim > 1:
            y_pred_train = y_pred_train.ravel()
        rmse_train = float(rmse(y_train_global, y_pred_train))
        
        y_pred_test = network.predict(X_test_global)
        if y_pred_test.ndim > 1:
            y_pred_test = y_pred_test.ravel()
        rmse_test = float(rmse(y_test_global, y_pred_test))
        
        # Calculate metrics
        improvement = ((rmse_initial - rmse_train) / rmse_initial * 100)
        gap = rmse_test - rmse_train
        
        return jsonify({
            'success': True,
            'network_info': {
                'layer_sizes': network_info['layer_sizes'],
                'total_parameters': int(network_info['total_parameters']),
                'activation_functions': network_info['activation_functions']
            },
            'pso_params': {
                'swarm_size': swarm_size,
                'num_informants': num_informants,
                'max_iterations': max_iterations,
                'bounds': bounds
            },
            'metrics': {
                'rmse_initial': rmse_initial,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'improvement_percent': improvement,
                'train_test_gap': gap,
                'generalization_status': 'Good' if abs(gap) < 2.0 else 'Warning'
            },
            'convergence_history': [float(x) for x in fitness_history],
            'iterations_count': len(fitness_history)
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400


@app.route('/api/suggested-configs')
def suggested_configs():
    """Get suggested network configurations."""
    configs = [
        {
            'name': 'Small Network',
            'layer_sizes': [8, 64, 32, 1],
            'activation_functions': ['relu', 'relu', 'linear'],
            'swarm_size': 25,
            'max_iterations': 50,
            'description': 'Small 3-layer network, fast computation'
        },
        {
            'name': 'Medium Network',
            'layer_sizes': [8, 128, 64, 32, 1],
            'activation_functions': ['relu', 'relu', 'relu', 'linear'],
            'swarm_size': 30,
            'max_iterations': 60,
            'description': 'Medium 4-layer network, balanced'
        },
        {
            'name': 'Deep Network',
            'layer_sizes': [8, 200, 150, 100, 50, 1],
            'activation_functions': ['relu', 'relu', 'relu', 'relu', 'linear'],
            'swarm_size': 35,
            'max_iterations': 70,
            'description': 'Deep 5-layer network, high complexity'
        },
        {
            'name': 'Large Network',
            'layer_sizes': [8, 300, 200, 128, 64, 32, 1],
            'activation_functions': ['relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
            'swarm_size': 40,
            'max_iterations': 80,
            'description': 'Large 6-layer network, very deep'
        }
    ]
    return jsonify(configs)


@app.template_filter('number')
def number_filter(value):
    """Format numbers for display."""
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)
    return str(value)


if __name__ == '__main__':
    if DATA_LOADED:
        print("Data loaded successfully!")
        print(f"Training samples: {X_train_global.shape[0]}")
        print(f"Test samples: {X_test_global.shape[0]}")
        print(f"Features: {X_train_global.shape[1]}")
        app.run(debug=True, port=5000)
    else:
        print("Error: Data file not found!")
        print(f"Expected: {npz_path}")
