#!/usr/bin/env python3
"""
PSO-ANN Trainer Module: Utilities for optimizing ANN weights using PSO.

Provides helper functions for:
- Data loading and conversion
- RMSE calculation
- Fitness function creation for PSO
- Integration of PSO and ANN optimization
"""
import numpy as np
from ann import NeuralNetwork
from pso import PSO


def to_numpy(arr):
    """Convert loaded arrays (possibly object dtype) to numeric numpy arrays."""
    arr = np.asarray(arr)
    if arr.dtype != object:
        return arr
    try:
        stacked = np.vstack(arr.tolist())
        return stacked
    except Exception:
        return np.array(arr.tolist(), dtype=float)


def rmse(y_true, y_pred):
    """Calculate RMSE between true and predicted values."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_fitness_function(network, X_train, y_train):
    """
    Create a fitness function for PSO.
    PSO minimizes this function by adjusting network weights.
    
    Args:
        network (NeuralNetwork): The neural network to optimize
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        
    Returns:
        callable: Fitness function that takes a parameter vector and returns RMSE
    """
    def fitness_function(parameters):
        # Set network parameters from the parameter vector (from PSO particle)
        network.set_parameters(parameters)
        
        # Forward pass
        y_pred = network.predict(X_train)
        
        # Flatten if needed
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        
        # Calculate and return RMSE (fitness to minimize)
        return rmse(y_train, y_pred)
    
    return fitness_function
