# PSO-ANN: Neural Network Optimization with Particle Swarm Optimization

A project demonstrating Particle Swarm Optimization for training Artificial Neural Networks on the UCI Concrete Compressive Strength dataset.

## Features

- **PSO Algorithm 39** with dynamic informants for neural network weight optimization
- **Interactive Web Interface** for running experiments and visualizing results
- **10 Pre-configured Experiments** with varying network architectures
- **Real-time Convergence Visualization** with 4-panel charts
- **Export Results** in CSV/JSON formats

## Quick Start

### Web Application (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask web app
python app.py

# Open browser to http://localhost:5000
```

### Command Line Demo

```bash
# Run a single PSO-ANN optimization
python src/main.py
```

### Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# - src/data_loader.ipynb (data preprocessing)
# - src/experiment_runner.ipynb (batch experiments)
# - src/run_pso_example.ipynb (PSO examples)
```

## Dataset

**UCI Concrete Compressive Strength Dataset**
- 1,030 samples
- 8 input features (cement, water, age, etc.)
- Target: Compressive strength (MPa)
- Split: 70% training (721 samples), 30% testing (309 samples)

## Technologies

- **Python 3.9+**
- **NumPy, Pandas** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **Scikit-learn** - Data preprocessing and metrics
- **Flask** - Web interface
- **Jupyter** - Interactive notebooks

## Project Structure

```
├── app.py                    # Flask web application
├── src/
│   ├── ann.py               # Neural network implementation
│   ├── pso.py               # PSO Algorithm 39
│   ├── pso_ann_trainer.py   # PSO-ANN integration
│   ├── main.py              # Standalone demo
│   └── experiment_runner.ipynb  # Batch experiments
├── data/
│   ├── processed_data.npz   # Preprocessed dataset
│   └── scaler.joblib        # Feature scaler
├── results/
│   ├── experiment_results.csv
│   └── experiment_results.json
├── static/                  # CSS, JavaScript
└── templates/               # HTML templates
```

## Deployment

**Render (Cloud):**
- Push code to GitHub
- Connect repository to Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Environment: Python 3
- Render automatically detects PORT and handles deployment

**Local:**
- Runs on `http://localhost:5000`
- Data and results persist in `data/` and `results/` directories

## Results

- **Mean RMSE Improvement**: 63.67% across 10 experiments
- **Best Test RMSE**: 13.73 (Experiment 6)
- **Architectures Tested**: 4-6 hidden layers, 11K-135K parameters
- **Interactive 4-panel visualizations**: Training curves, predictions, residuals, convergence
- **Export**: Download results as CSV/JSON