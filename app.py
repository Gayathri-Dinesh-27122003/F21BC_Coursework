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
from pso_ann_trainer import (rmse, mae, create_fitness_function, to_numpy,
                              cross_validate_pso_ann, run_multiple_experiments)

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
        
        # New parameters for advanced features
        num_runs = int(params.get('num_runs', 1))  # Number of independent runs
        use_cross_validation = params.get('use_cross_validation', False)
        k_folds = int(params.get('k_folds', 5))
        
        # PSO acceleration coefficients
        w = float(params.get('w', 0.729))  # Inertia weight
        c1 = float(params.get('c1', 1.49445))  # Cognitive coefficient
        c2 = float(params.get('c2', 1.49445))  # Social coefficient
        
        # Validate activation functions
        valid_activations = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'linear']
        for act in activation_functions:
            if act not in valid_activations:
                return jsonify({'error': f'Invalid activation function: {act}. Valid options: {", ".join(valid_activations)}'}), 400
        
        # Validate layer_sizes matches number of activation functions
        if len(layer_sizes) - 1 != len(activation_functions):
            return jsonify({'error': 'Number of activation functions must be (number of layers - 1)'}), 400
        
        # Handle cross-validation mode
        if use_cross_validation:
            # Create temporary network to get parameter count
            temp_network = NeuralNetwork(layer_sizes, activation_functions)
            temp_network_info = temp_network.get_network_info()
            
            cv_results = cross_validate_pso_ann(
                np.vstack([X_train_global, X_test_global]),
                np.concatenate([y_train_global, y_test_global]),
                layer_sizes, activation_functions,
                swarm_size, num_informants, max_iterations, bounds, k_folds
            )
            
            return jsonify({
                'success': True,
                'mode': 'cross_validation',
                'network_info': {
                    'layer_sizes': layer_sizes,
                    'total_parameters': int(temp_network_info['total_parameters']),
                    'activation_functions': activation_functions
                },
                'pso_params': {
                    'swarm_size': swarm_size,
                    'num_informants': num_informants,
                    'max_iterations': max_iterations,
                    'bounds': bounds,
                    'w': w, 'c1': c1, 'c2': c2
                },
                'cross_validation': cv_results
            })
        
        # Handle multiple runs mode
        if num_runs > 1:
            # Create temporary network to get parameter count
            temp_network = NeuralNetwork(layer_sizes, activation_functions)
            temp_network_info = temp_network.get_network_info()
            
            multi_run_results = run_multiple_experiments(
                X_train_global, y_train_global, X_test_global, y_test_global,
                layer_sizes, activation_functions,
                swarm_size, num_informants, max_iterations, bounds, num_runs
            )
            
            return jsonify({
                'success': True,
                'mode': 'multiple_runs',
                'network_info': {
                    'layer_sizes': layer_sizes,
                    'total_parameters': int(temp_network_info['total_parameters']),
                    'activation_functions': activation_functions
                },
                'pso_params': {
                    'swarm_size': swarm_size,
                    'num_informants': num_informants,
                    'max_iterations': max_iterations,
                    'bounds': bounds,
                    'w': w, 'c1': c1, 'c2': c2
                },
                'multiple_runs': multi_run_results
            })
        
        # Standard single run
        
        # Create network
        network = NeuralNetwork(layer_sizes, activation_functions)
        network_info = network.get_network_info()
        
        # Calculate initial RMSE and MAE
        y_pred_initial = network.predict(X_train_global)
        if y_pred_initial.ndim > 1:
            y_pred_initial = y_pred_initial.ravel()
        rmse_initial = float(rmse(y_train_global, y_pred_initial))
        mae_initial = float(mae(y_train_global, y_pred_initial))
        
        # Create fitness function
        fitness_func = create_fitness_function(network, X_train_global, y_train_global)
        
        # Run PSO
        pso = PSO(
            objective_function=fitness_func,
            dimension=network_info['total_parameters'],
            swarm_size=swarm_size,
            num_informants=num_informants,
            w=w,
            c1=c1,
            c2=c2,
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
        mae_train = float(mae(y_train_global, y_pred_train))
        
        y_pred_test = network.predict(X_test_global)
        if y_pred_test.ndim > 1:
            y_pred_test = y_pred_test.ravel()
        rmse_test = float(rmse(y_test_global, y_pred_test))
        mae_test = float(mae(y_test_global, y_pred_test))
        
        # Calculate metrics
        improvement = ((rmse_initial - rmse_train) / rmse_initial * 100)
        gap = rmse_test - rmse_train
        
        return jsonify({
            'success': True,
            'mode': 'single_run',
            'network_info': {
                'layer_sizes': network_info['layer_sizes'],
                'total_parameters': int(network_info['total_parameters']),
                'activation_functions': network_info['activation_functions']
            },
            'pso_params': {
                'swarm_size': swarm_size,
                'num_informants': num_informants,
                'max_iterations': max_iterations,
                'bounds': bounds,
                'w': w,
                'c1': c1,
                'c2': c2
            },
            'metrics': {
                'rmse_initial': rmse_initial,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'mae_initial': mae_initial,
                'mae_train': mae_train,
                'mae_test': mae_test,
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
        
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
