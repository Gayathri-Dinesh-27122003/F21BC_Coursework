import numpy as np
from pathlib import Path
import joblib

from ann import NeuralNetwork
from pso import PSO
from pso_ann_trainer import rmse, create_fitness_function, to_numpy


def main():
    """Optimize ANN weights using PSO on concrete dataset."""
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / 'data'

    print("PSO-ANN INTEGRATION DEMO")
    print("Optimizing Neural Network Weights using Particle Swarm Optimization")
    
    # Load processed data
    print("\nLoading Concrete Compressive Strength Dataset")
    npz_path = data_dir / 'processed_data.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X_train = to_numpy(data['X_train'])
    X_test = to_numpy(data['X_test'])
    y_train = to_numpy(data['y_train']).ravel()
    y_test = to_numpy(data['y_test']).ravel()

    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    n_features = X_train.shape[1]

    # Create neural network
    print("\nCreating Neural Network Architecture")
    network = NeuralNetwork(
        layer_sizes=[n_features, 64, 32, 1],
        activation_functions=['relu', 'relu', 'linear']
    )
    network_info = network.get_network_info()
    
    print(f"  Architecture: {network_info['layer_sizes']}")
    print(f"  Total parameters: {network_info['total_parameters']}")
    print(f"  Activation functions: {network_info['activation_functions']}")

    # Calculate initial (random weights) RMSE
    print("\nCalculating Initial Performance (Random Weights)")
    y_pred_initial = network.predict(X_train)
    if y_pred_initial.ndim > 1 and y_pred_initial.shape[1] == 1:
        y_pred_initial = y_pred_initial.ravel()
    rmse_initial = rmse(y_train, y_pred_initial)
    print(f"  Initial RMSE (training): {rmse_initial:.6f}")

    # Create fitness function for PSO
    print("\nSetting up PSO Optimization")
    fitness_func = create_fitness_function(network, X_train, y_train)
    print(f"  Fitness function created (minimizes RMSE)")

    # Create and run PSO
    print("\nRunning Particle Swarm Optimization")
    pso = PSO(
        objective_function=fitness_func,
        dimension=network_info['total_parameters'],
        swarm_size=30,
        num_informants=3,
        w=0.729,
        c1=1.49445,
        c2=1.49445,
        bounds=(-5.0, 5.0),
        max_iterations=50
    )

    best_params, best_fitness, fitness_history = pso.optimize(verbose=True)

    # Apply best parameters to network
    print("\nApplying Optimized Parameters to Network...")
    network.set_parameters(best_params)

    # Evaluate on training data
    y_pred_train = network.predict(X_train)
    if y_pred_train.ndim > 1 and y_pred_train.shape[1] == 1:
        y_pred_train = y_pred_train.ravel()
    rmse_train = rmse(y_train, y_pred_train)

    # Evaluate on test data
    y_pred_test = network.predict(X_test)
    if y_pred_test.ndim > 1 and y_pred_test.shape[1] == 1:
        y_pred_test = y_pred_test.ravel()
    rmse_test = rmse(y_test, y_pred_test)

    # Print final results
    print("\nFINAL RESULTS")
    print(f"\nPerformance Metrics:")
    print(f"  Initial RMSE (random weights): {rmse_initial:.6f}")
    print(f"  Final RMSE (training set): {rmse_train:.6f}")
    print(f"  Final RMSE (test set): {rmse_test:.6f}")
    
    improvement = ((rmse_initial - rmse_train) / rmse_initial * 100)
    print(f"\nOptimization Results:")
    print(f"  RMSE Improvement: {improvement:.2f}%")
    print(f"  PSO Iterations: {len(fitness_history)}")
    print(f"  Best Fitness Value: {best_fitness:.6f}")
    
    print(f"\nGeneralization:")
    gap = rmse_test - rmse_train
    print(f"  Train-Test Gap: {gap:.6f}")
    if gap < 0.5:
        print(f"  Status: Good generalization")
    else:
        print(f"  Status: Possible overfitting")

    # Optional: Save results
    summary = {
        'rmse_initial': rmse_initial,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'improvement_pct': improvement,
        'best_params': best_params,
        'fitness_history': fitness_history,
        'network_info': network_info,
        'pso_summary': pso.get_optimization_summary()
    }
    
    print("\nResults Summary saved (can be exported with joblib.dump)")
    return summary


if __name__ == '__main__':
    print("\nInitializing PSO-ANN Integration Demo...\n")
    summary = main()
    print("\nDemo completed successfully!\n")
