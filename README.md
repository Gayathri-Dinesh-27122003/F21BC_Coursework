PSO-ANN: Neural Network Optimization with Particle Swarm Optimization

A project demonstrating Particle Swarm Optimization for training Neural Networks on the Concrete Compressive Strength dataset.

Quick Start

Web Application (Recommended):

pip install -r requirements.txt
python app.py
# Open http://localhost:5000

Command Line Demo

python src/main.py

Docker

docker build -t pso-ann .
docker run -p 5000:5000 pso-ann

Technologies

Python 3.9
NumPy, Pandas, Matplotlib
Scikit-learn
Flask (web interface)
Jupyter Notebooks

Outputs

Interactive 4-panel visualizations
CSV/JSON results export
Model performance metrics
Convergence analysis