// Global variables
let suggestedConfigs = [];
let lastResults = null;
let convergenceChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadDatasetInfo();
    await loadSuggestedConfigs();
    setupLayerPreview();
    setupAdvancedControls();
    // Initialize preview and activation controls on first load
    updateLayerPreview();
    updateActivationControls();
});

// Setup advanced controls listeners
function setupAdvancedControls() {
    // Toggle k-folds input based on cross-validation checkbox
    const cvCheckbox = document.getElementById('useCrossValidation');
    const kFoldsGroup = document.getElementById('kFoldsGroup');
    
    cvCheckbox.addEventListener('change', () => {
        kFoldsGroup.style.display = cvCheckbox.checked ? 'block' : 'none';
    });
}

// Load dataset information
async function loadDatasetInfo() {
    try {
        const response = await fetch('/api/data-info');
        const data = await response.json();

        const html = `
            <p><strong>Training Samples:</strong> ${data.train_samples.toLocaleString()}</p>
            <p><strong>Test Samples:</strong> ${data.test_samples.toLocaleString()}</p>
            <p><strong>Features:</strong> ${data.features}</p>
            <p><strong>Target Range:</strong> ${data.target_min.toFixed(2)} - ${data.target_max.toFixed(2)}</p>
            <p><strong>Target Mean:</strong> ${data.target_mean.toFixed(2)} ± ${data.target_std.toFixed(2)}</p>
        `;
        document.getElementById('datasetInfo').innerHTML = html;
    } catch (error) {
        console.error('Error loading dataset info:', error);
        document.getElementById('datasetInfo').innerHTML = '<p class="placeholder">Error loading dataset</p>';
    }
}

// Load suggested configurations
async function loadSuggestedConfigs() {
    try {
        const response = await fetch('/api/suggested-configs');
        suggestedConfigs = await response.json();
    } catch (error) {
        console.error('Error loading suggested configs:', error);
    }
}

// Load template configuration
function loadTemplate(index) {
    if (index >= 0 && index < suggestedConfigs.length) {
        const config = suggestedConfigs[index];
        const hidden = config.layer_sizes.slice(1, -1); // Remove input and output layers

        document.getElementById('hiddenLayers').value = hidden.join(',');
        document.getElementById('swarmSize').value = config.swarm_size;
        document.getElementById('maxIterations').value = config.max_iterations;

        updateLayerPreview();
        updateActivationControls();
        // Ensure activation controls update when loading a template programmatically
        updateActivationControls();

        const status = document.getElementById('status');
        status.textContent = `Loaded: ${config.name} - ${config.description}`;
        status.className = 'status success';
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 3000);
    }
}

// Setup layer preview update
function setupLayerPreview() {
    const hiddenInput = document.getElementById('hiddenLayers');
    const updateBoth = () => {
        updateLayerPreview();
        updateActivationControls();
    };

    // Listen for typing and for change events (e.g., paste, programmatic changes)
    hiddenInput.addEventListener('input', updateBoth);
    hiddenInput.addEventListener('change', updateBoth);
}

// Update layer preview
function updateLayerPreview() {
    const hiddenText = document.getElementById('hiddenLayers').value.trim();
    const preview = document.getElementById('layerPreview');

    if (hiddenText === '') {
        preview.classList.remove('visible');
        return;
    }

    try {
        const hidden = hiddenText.split(',').map(x => parseInt(x.trim()));

        if (hidden.some(x => isNaN(x) || x <= 0)) {
            throw new Error('Invalid layer sizes');
        }

        const layers = [8, ...hidden, 1];
        const architecture = layers.map(l => `[${l}]`).join(' -> ');
        const totalParams = calculateTotalParameters(layers);

        preview.textContent = `Architecture: ${architecture}\nTotal Parameters: ${totalParams.toLocaleString()}`;
        preview.classList.add('visible');
    } catch (error) {
        preview.textContent = 'Invalid layer configuration';
        preview.classList.add('visible');
    }
}

// Update activation function controls based on number of layers
function updateActivationControls() {
    const hiddenText = document.getElementById('hiddenLayers').value.trim();
    const container = document.getElementById('activationContainer');
    
    if (hiddenText === '') {
        container.innerHTML = '';
        return;
    }

    try {
        // Parse and filter out invalid entries (handles extra commas/spaces)
        const hidden = hiddenText.split(',')
            .map(x => parseInt(x.trim()))
            .filter(n => !isNaN(n) && n > 0);
        const numLayers = hidden.length + 1; // +1 for output layer

        const activationOptions = [
            { value: 'relu', label: 'ReLU' },
            { value: 'tanh', label: 'Tanh' },
            { value: 'sigmoid', label: 'Sigmoid' },
            { value: 'elu', label: 'ELU' },
            { value: 'selu', label: 'SELU' },
            { value: 'linear', label: 'Linear' }
        ];

        let html = '';

        // Hidden layers
        for (let i = 0; i < hidden.length; i++) {
            html += `
                <div class="form-group">
                    <label>Hidden Layer ${i + 1} (${hidden[i]} units)</label>
                    <select id="activation_${i}" class="layer-activation">
                        ${activationOptions.map(opt => 
                            `<option value="${opt.value}" ${opt.value === 'relu' ? 'selected' : ''}>${opt.label}</option>`
                        ).join('')}
                    </select>
                </div>
            `;
        }

        // Output layer
        html += `
            <div class="form-group">
                <label>Output Layer (1 unit)</label>
                <select id="activation_output" class="layer-activation">
                    <option value="linear" selected>Linear</option>
                    <option value="relu">ReLU</option>
                    <option value="tanh">Tanh</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="elu">ELU</option>
                    <option value="selu">SELU</option>
                </select>
                <small>Linear recommended for regression</small>
            </div>
        `;

        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = '';
    }
}

// Calculate total parameters (rough estimate)
function calculateTotalParameters(layers) {
    let total = 0;
    for (let i = 0; i < layers.length - 1; i++) {
        // weights: input * output, biases: output
        total += layers[i] * layers[i + 1] + layers[i + 1];
    }
    return total;
}

// Parse layer sizes from input
function parseLayerSizes() {
    const input = document.getElementById('hiddenLayers').value.trim();

    if (input === '') {
        return [8, 32, 1]; // Default
    }

    try {
        const hidden = input.split(',').map(x => {
            const num = parseInt(x.trim());
            if (isNaN(num) || num <= 0) throw new Error('Invalid layer size');
            return num;
        });
        return [8, ...hidden, 1];
    } catch (error) {
        throw new Error('Invalid layer configuration: ' + error.message);
    }
}

// Generate activation functions from individual layer selections
function getActivationFunctions() {
    const activations = [];
    
    // Get all layer activation selects
    const hiddenText = document.getElementById('hiddenLayers').value.trim();
    const hidden = hiddenText.split(',').map(x => parseInt(x.trim()));
    const numHiddenLayers = hidden.length;
    
    // Collect hidden layer activations
    for (let i = 0; i < numHiddenLayers; i++) {
        const selectId = `activation_${i}`;
        const select = document.getElementById(selectId);
        if (select) {
            activations.push(select.value);
        } else {
            activations.push('relu'); // Default fallback
        }
    }
    
    // Collect output layer activation
    const outputSelect = document.getElementById('activation_output');
    if (outputSelect) {
        activations.push(outputSelect.value);
    } else {
        activations.push('linear'); // Default fallback
    }
    
    return activations;
}

// Run experiment
async function runExperiment() {
    const runButton = document.getElementById('runButton');
    const status = document.getElementById('status');

    try {
        runButton.disabled = true;
        status.className = 'status loading';
        status.textContent = 'Running experiment...';

        // Parse inputs
        const layerSizes = parseLayerSizes();
        const activations = getActivationFunctions();
        const swarmSize = parseInt(document.getElementById('swarmSize').value);
        const numInformants = parseInt(document.getElementById('numInformants').value);
        const maxIterations = parseInt(document.getElementById('maxIterations').value);
        const bounds = [-1.0, 1.0]; // Fixed bounds
        
        // Advanced parameters
        const numRuns = parseInt(document.getElementById('numRuns').value);
        const useCrossValidation = document.getElementById('useCrossValidation').checked;
        const kFolds = parseInt(document.getElementById('kFolds').value);
        const w = parseFloat(document.getElementById('inertiaWeight').value);
        const c1 = parseFloat(document.getElementById('cognitiveCoeff').value);
        const c2 = parseFloat(document.getElementById('socialCoeff').value);

        // Validate inputs
        if (swarmSize < 5 || swarmSize > 100) {
            throw new Error('Swarm size must be between 5 and 100');
        }
        if (numInformants < 1 || numInformants > swarmSize) {
            throw new Error('Informants must be between 1 and swarm size');
        }
        if (maxIterations < 10 || maxIterations > 200) {
            throw new Error('Iterations must be between 10 and 200');
        }

        // Send request
        const response = await fetch('/api/run-experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                layer_sizes: layerSizes,
                activation_functions: activations,
                swarm_size: swarmSize,
                num_informants: numInformants,
                max_iterations: maxIterations,
                bounds: bounds,
                num_runs: numRuns,
                use_cross_validation: useCrossValidation,
                k_folds: kFolds,
                w: w,
                c1: c1,
                c2: c2
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Experiment failed');
        }

        lastResults = result;

        // Update results
        displayResults(result);

        status.className = 'status success';
        status.textContent = 'Experiment completed successfully!';
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 3000);
    } catch (error) {
        console.error('Error:', error);
        status.className = 'status error';
        status.textContent = 'Error: ' + error.message;
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 5000);
    } finally {
        runButton.disabled = false;
    }
}

// Display results
function displayResults(result) {
    const metricsBox = document.getElementById('metricsBox');
    
    // Handle different modes
    if (result.mode === 'cross_validation') {
        displayCrossValidationResults(result);
        return;
    } else if (result.mode === 'multiple_runs') {
        displayMultipleRunsResults(result);
        return;
    }
    
    // Standard single run display
    const metrics = result.metrics;

    let metricsHtml = `
        <div class="metric">
            <div class="metric-label">Initial RMSE</div>
            <div class="metric-value">${metrics.rmse_initial.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Final RMSE (Train)</div>
            <div class="metric-value">${metrics.rmse_train.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Final RMSE (Test)</div>
            <div class="metric-value">${metrics.rmse_test.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">MAE (Train)</div>
            <div class="metric-value">${metrics.mae_train.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">MAE (Test)</div>
            <div class="metric-value">${metrics.mae_test.toFixed(4)}</div>
        </div>
        <div class="metric success">
            <div class="metric-label">Improvement</div>
            <div class="metric-value">${metrics.improvement_percent.toFixed(2)}%</div>
        </div>
        <div class="metric ${metrics.train_test_gap > 2 ? 'warning' : 'success'}">
            <div class="metric-label">Train-Test Gap</div>
            <div class="metric-value">${metrics.train_test_gap.toFixed(4)}</div>
        </div>
        <div class="metric ${metrics.generalization_status === 'Good' ? 'success' : 'warning'}">
            <div class="metric-label">Generalization</div>
            <div class="metric-value">${metrics.generalization_status}</div>
        </div>
    `;

    metricsBox.innerHTML = metricsHtml;

    // Show convergence chart for single run
    const chartSection = document.querySelector('.chart-section');
    if (chartSection) {
        chartSection.style.display = 'block';
    }
    
    // Update convergence chart
    updateConvergenceChart(result.convergence_history);

    // Display network summary
    displayNetworkSummary(result);

    // Show export button
    document.getElementById('exportButton').style.display = 'block';
}

// Display cross-validation results
function displayCrossValidationResults(result) {
    const metricsBox = document.getElementById('metricsBox');
    const cv = result.cross_validation;
    
    let metricsHtml = `
        <div class="metric success">
            <div class="metric-label">Cross-Validation Mode</div>
            <div class="metric-value">${cv.k_folds}-Fold CV</div>
        </div>
        <div class="metric">
            <div class="metric-label">RMSE (Mean ± Std)</div>
            <div class="metric-value">${cv.rmse_mean.toFixed(4)} ± ${cv.rmse_std.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">MAE (Mean ± Std)</div>
            <div class="metric-value">${cv.mae_mean.toFixed(4)} ± ${cv.mae_std.toFixed(4)}</div>
        </div>
        <div class="metric success">
            <div class="metric-label">Improvement (Mean ± Std)</div>
            <div class="metric-value">${cv.improvement_mean.toFixed(2)}% ± ${cv.improvement_std.toFixed(2)}%</div>
        </div>
    `;
    
    // Add fold-by-fold results
    cv.fold_results.forEach(fold => {
        metricsHtml += `
            <div class="metric">
                <div class="metric-label">Fold ${fold.fold} RMSE</div>
                <div class="metric-value">${fold.rmse_val.toFixed(4)}</div>
            </div>
        `;
    });
    
    metricsBox.innerHTML = metricsHtml;
    
    // Display convergence chart for each fold
    const chartSection = document.querySelector('.chart-section');
    if (chartSection) {
        let chartsHtml = '<h2>Convergence History by Fold</h2>';
        
        cv.fold_results.forEach((fold, idx) => {
            chartsHtml += `
                <div style="margin-bottom: 25px; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.1em; font-weight: 600;">Fold ${fold.fold} - RMSE: ${fold.rmse_val.toFixed(4)} | Improvement: ${fold.improvement_percent.toFixed(2)}%</h3>
                    <div style="background: white; padding: 15px; border-radius: 5px;">
                        <canvas id="convergenceChart${idx}" width="800" height="300"></canvas>
                    </div>
                </div>
            `;
        });
        
        chartSection.innerHTML = chartsHtml;
        chartSection.style.display = 'block';
        
        // Draw each chart
        cv.fold_results.forEach((fold, idx) => {
            const canvas = document.getElementById(`convergenceChart${idx}`);
            if (canvas && fold.convergence_history) {
                drawConvergenceChart(canvas, fold.convergence_history);
            }
        });
    }
    
    displayNetworkSummary(result);
    document.getElementById('exportButton').style.display = 'block';
}

// Display multiple runs results
function displayMultipleRunsResults(result) {
    const metricsBox = document.getElementById('metricsBox');
    const mr = result.multiple_runs;
    
    let metricsHtml = `
        <div class="metric success">
            <div class="metric-label">Multiple Runs</div>
            <div class="metric-value">${mr.num_runs} Runs</div>
        </div>
        <div class="metric">
            <div class="metric-label">RMSE (Mean ± Std)</div>
            <div class="metric-value">${mr.rmse_test_mean.toFixed(4)} ± ${mr.rmse_test_std.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">RMSE Range</div>
            <div class="metric-value">${mr.rmse_test_min.toFixed(4)} - ${mr.rmse_test_max.toFixed(4)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">MAE (Mean ± Std)</div>
            <div class="metric-value">${mr.mae_test_mean.toFixed(4)} ± ${mr.mae_test_std.toFixed(4)}</div>
        </div>
        <div class="metric success">
            <div class="metric-label">Improvement (Mean ± Std)</div>
            <div class="metric-value">${mr.improvement_mean.toFixed(2)}% ± ${mr.improvement_std.toFixed(2)}%</div>
        </div>
    `;
    
    // Add individual run results (first 6)
    mr.run_results.slice(0, 6).forEach(run => {
        metricsHtml += `
            <div class="metric">
                <div class="metric-label">Run ${run.run} RMSE</div>
                <div class="metric-value">${run.rmse_test.toFixed(4)}</div>
            </div>
        `;
    });
    
    metricsBox.innerHTML = metricsHtml;
    
    // Display convergence chart for each run
    const chartSection = document.querySelector('.chart-section');
    if (chartSection) {
        let chartsHtml = '<h2>Convergence History by Run</h2>';
        
        runs.run_results.forEach((run, idx) => {
            chartsHtml += `
                <div style="margin-bottom: 25px; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.1em; font-weight: 600;">Run ${run.run} - RMSE: ${run.rmse_test.toFixed(4)} | Improvement: ${run.improvement_percent.toFixed(2)}%</h3>
                    <div style="background: white; padding: 15px; border-radius: 5px;">
                        <canvas id="convergenceChart${idx}" width="800" height="300"></canvas>
                    </div>
                </div>
            `;
        });
        
        chartSection.innerHTML = chartsHtml;
        chartSection.style.display = 'block';
        
        // Draw each chart
        runs.run_results.forEach((run, idx) => {
            const canvas = document.getElementById(`convergenceChart${idx}`);
            if (canvas && run.convergence_history) {
                drawConvergenceChart(canvas, run.convergence_history);
            }
        });
    }
    
    displayNetworkSummary(result);
    document.getElementById('exportButton').style.display = 'block';
}

// Helper function to draw convergence chart on a specific canvas
function drawConvergenceChart(canvas, history) {
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: history.length}, (_, i) => i),
            datasets: [
                {
                    label: 'Best Fitness (RMSE)',
                    data: history,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#333',
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#666',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'RMSE',
                        color: '#333'
                    }
                },
                x: {
                    ticks: {
                        color: '#666',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'Iteration',
                        color: '#333'
                    }
                }
            }
        }
    });
}

// Update convergence chart
function updateConvergenceChart(history) {
    const ctx = document.getElementById('convergenceChart').getContext('2d');

    if (convergenceChart) {
        convergenceChart.destroy();
    }

    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: history.length}, (_, i) => i),
            datasets: [
                {
                    label: 'Best Fitness (RMSE)',
                    data: history,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#333'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#666'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    ticks: {
                        color: '#666'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Display network summary
function displayNetworkSummary(result) {
    const summary = result.network_info;
    
    // Build network info section
    let networkInfoHtml = '';
    if (summary.layer_sizes) {
        networkInfoHtml += `<p><strong>Layer Sizes:</strong> ${summary.layer_sizes.join(' -> ')}</p>`;
    }
    if (summary.total_parameters !== undefined) {
        networkInfoHtml += `<p><strong>Total Parameters:</strong> ${summary.total_parameters.toLocaleString()}</p>`;
    }
    if (summary.activation_functions) {
        networkInfoHtml += `<p><strong>Activation Functions:</strong> ${summary.activation_functions.join(', ')}</p>`;
    }
    
    // Build PSO config section
    let psoConfigHtml = '';
    if (result.pso_params) {
        psoConfigHtml = `
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
            <p><strong>PSO Configuration:</strong></p>
            <p style="margin-left: 15px;">
                Swarm: ${result.pso_params.swarm_size} particles<br>
                Informants: ${result.pso_params.num_informants}<br>
                Iterations: ${result.pso_params.max_iterations}<br>
                Bounds: [${result.pso_params.bounds[0]}, ${result.pso_params.bounds[1]}]<br>
                ${result.pso_params.w ? `w=${result.pso_params.w.toFixed(3)}, c1=${result.pso_params.c1.toFixed(3)}, c2=${result.pso_params.c2.toFixed(3)}` : ''}
            </p>
        `;
    }
    
    document.getElementById('networkSummary').innerHTML = networkInfoHtml + psoConfigHtml;
}

// Export results as JSON
function exportResults() {
    if (!lastResults) return;

    const dataStr = JSON.stringify(lastResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pso-ann-results-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}
