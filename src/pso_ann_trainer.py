import numpy as np
from ann import NeuralNetwork
from pso import PSO


def to_numpy(arr):
    """Convert loaded arrays to numeric numpy arrays."""
    arr = np.asarray(arr)
    if arr.dtype != object:
        return arr
    try:
        stacked = np.vstack(arr.tolist())
        return stacked
    except Exception:
        return np.array(arr.tolist(), dtype=float)


def rmse(y_true, y_pred):
    """Calculate RMSE """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Calculate MAE """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(np.abs(y_true - y_pred))


def create_fitness_function(network, X_train, y_train):
    
    def fitness_function(parameters):
        # Set network parameters from PSO particle
        network.set_parameters(parameters)
        
        # Forward pass
        y_pred = network.predict(X_train)
        
        # Flatten 
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        
        # Calculate and return RMSE 
        return rmse(y_train, y_pred)
    
    return fitness_function


def cross_validate_pso_ann(X, y, layer_sizes, activation_functions, 
                           swarm_size=30, num_informants=3, max_iterations=50, 
                           bounds=(-5.0, 5.0), k_folds=5):
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create network
        network = NeuralNetwork(layer_sizes, activation_functions)
        
        # Calculate initial RMSE
        y_pred_init = network.predict(X_val_fold)
        rmse_init = rmse(y_val_fold, y_pred_init)
        
        # Create fitness function
        fitness_fn = create_fitness_function(network, X_train_fold, y_train_fold)
        
        # Run PSO
        pso = PSO(
            objective_function=fitness_fn,
            dimension=network.get_parameter_count(),
            swarm_size=swarm_size,
            num_informants=num_informants,
            max_iterations=max_iterations,
            bounds=bounds
        )
        
        best_params, best_fitness, convergence_history = pso.optimize(verbose=False)
        network.set_parameters(best_params)
        
        # Evaluate on validation fold
        y_pred_train = network.predict(X_train_fold)
        y_pred_val = network.predict(X_val_fold)
        
        rmse_val_final = rmse(y_val_fold, y_pred_val)
        improvement_pct = ((rmse_init - rmse_val_final) / rmse_init) * 100
        
        fold_results.append({
            'fold': fold_idx + 1,
            'rmse_initial': rmse_init,
            'rmse_train': rmse(y_train_fold, y_pred_train),
            'rmse_val': rmse_val_final,
            'mae_train': mae(y_train_fold, y_pred_train),
            'mae_val': mae(y_val_fold, y_pred_val),
            'improvement_percent': improvement_pct,
            'convergence_history': convergence_history
        })
    
    # Calculate statistics across folds
    rmse_vals = [f['rmse_val'] for f in fold_results]
    mae_vals = [f['mae_val'] for f in fold_results]
    improvements = [f['improvement_percent'] for f in fold_results]
    
    return {
        'fold_results': fold_results,
        'rmse_mean': np.mean(rmse_vals),
        'rmse_std': np.std(rmse_vals),
        'mae_mean': np.mean(mae_vals),
        'mae_std': np.std(mae_vals),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'k_folds': k_folds
    }


def run_multiple_experiments(X_train, y_train, X_test, y_test,
                             layer_sizes, activation_functions,
                             swarm_size=30, num_informants=3, 
                             max_iterations=50, bounds=(-5.0, 5.0),
                             num_runs=10):
    
    run_results = []
    
    for run_idx in range(num_runs):
        # Create network
        network = NeuralNetwork(layer_sizes, activation_functions)
        
        # Initial evaluation
        y_pred_init = network.predict(X_test)
        rmse_init = rmse(y_test, y_pred_init)
        mae_init = mae(y_test, y_pred_init)
        
        # Create fitness and run PSO
        fitness_fn = create_fitness_function(network, X_train, y_train)
        pso = PSO(
            objective_function=fitness_fn,
            dimension=network.get_parameter_count(),
            swarm_size=swarm_size,
            num_informants=num_informants,
            max_iterations=max_iterations,
            bounds=bounds
        )
        
        best_params, best_fitness, history = pso.optimize(verbose=False)
        network.set_parameters(best_params)
        
        # Final evaluation
        y_pred_train = network.predict(X_train)
        y_pred_test = network.predict(X_test)
        
        run_results.append({
            'run': run_idx + 1,
            'rmse_initial': rmse_init,
            'mae_initial': mae_init,
            'rmse_train': rmse(y_train, y_pred_train),
            'rmse_test': rmse(y_test, y_pred_test),
            'mae_train': mae(y_train, y_pred_train),
            'mae_test': mae(y_test, y_pred_test),
            'improvement_percent': ((rmse_init - rmse(y_test, y_pred_test)) / rmse_init) * 100,
            'convergence_history': history
        })
    
    # Calculate statistics
    rmse_tests = [r['rmse_test'] for r in run_results]
    mae_tests = [r['mae_test'] for r in run_results]
    improvements = [r['improvement_percent'] for r in run_results]
    
    return {
        'run_results': run_results,
        'num_runs': num_runs,
        'rmse_test_mean': np.mean(rmse_tests),
        'rmse_test_std': np.std(rmse_tests),
        'rmse_test_min': np.min(rmse_tests),
        'rmse_test_max': np.max(rmse_tests),
        'mae_test_mean': np.mean(mae_tests),
        'mae_test_std': np.std(mae_tests),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements)
    }
