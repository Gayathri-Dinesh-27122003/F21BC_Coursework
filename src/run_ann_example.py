import numpy as np
from pathlib import Path
import joblib

# import your NeuralNetwork from ann
from ann import NeuralNetwork


def to_numpy(arr):
    
    # if it's already a numpy array of numeric dtype, return directly
    arr = np.asarray(arr)
    if arr.dtype != object:
        return arr

    # If object dtype, try to convert element-wise
    try:
        # Some saves may store a 1-D object array of rows; try vstack of tolist()
        stacked = np.vstack(arr.tolist())
        return stacked
    except Exception:
        # fallback to converting list
        return np.array(arr.tolist(), dtype=float)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / 'data'

    # load processed data (saved by data_loader notebook)
    npz_path = data_dir / 'processed_data.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    X_test = to_numpy(data['X_test'])
    y_test = to_numpy(data['y_test'])

    # Ensure targets are 1-D
    if y_test.ndim > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()

    n_features = X_test.shape[1]

    # Example network: [n_features, 64, 1] with relu hidden and linear output
    net = NeuralNetwork([n_features, 64, 1], ['relu', 'linear'])

    # Forward pass (inputs are expected to already be scaled)
    y_pred = net.predict(X_test)

    # Flatten prediction if shape (N,1)
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    score = rmse(y_test, y_pred)
    print(f"Example network RMSE (random init): {score:.6f}")


if __name__ == '__main__':
    main()
