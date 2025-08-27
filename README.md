# Deep Learning for Magnetotelluric Data Inversion

This project implements deep learning models for magnetotelluric (MT) data inversion, specifically for predicting subsurface resistivity profiles from apparent resistivity and phase measurements. The project supports multiple neural network architectures including LSTM, GRU, and Informer models.

## Overview

Magnetotelluric (MT) is a geophysical method that uses natural electromagnetic fields to investigate the electrical conductivity structure of the Earth's subsurface. This project uses deep learning to automate the inversion process, converting MT measurements (apparent resistivity and phase) into subsurface resistivity profiles.

## Requirements

```bash
pip install numpy matplotlib torch scikit-learn pandas scipy tqdm
```

### Dependencies
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- SciPy
- tqdm

## Installation

1. Clone or download the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure you have the required data structure (see Data Structure section)
```

### File Formats
- **EDI files**: Standard magnetotelluric data format containing frequencies, impedance components
- **Layered files**: Text files containing 1D resistivity model (depth, thickness, resistivity)

## Usage

### Command Line Interface

The script supports multiple modes and options:

```bash
# Train all models
python main.py --mode train --model all

# Train specific model
python main.py --mode train --model lstm

# Make predictions for a specific site
python main.py --mode predict --site 3D01 --model lstm

# Run prediction on all available sites
python main.py --mode predict --all-sites --model lstm

# Train and predict (full pipeline)
python main.py --mode both --model all

# Compare all trained models
python main.py --compare
```

### Command Line Arguments

- `--mode`: Operation mode (`train`, `predict`, `both`)
- `--model`: Model architecture (`lstm`, `gru`, `informer`, `all`)
- `--site`: Site name for prediction (default: `3D01`)
- `--seed`: Random seed for reproducibility (default: `99`)
- `--all-sites`: Process all available sites
- `--sites`: List of specific sites to process
- `--compare`: Generate comparison of all trained models

### Programmatic Usage

```python
from main import *

# Generate synthetic data
synthetic_data = generate_synthetic_data(num_samples=10000, num_periods=25)

# Train a model
model, scalers, metrics = train_model(synthetic_data, model_type='lstm', epochs=50)

# Make predictions
pred_profile, pred_depths, true_profile, true_depths = run_prediction('3D01', model_type='lstm')

# Compare models
comparison_df = compare_models(['lstm', 'gru', 'informer'])
```

## Model Architectures

### 1. LSTM (Long Short-Term Memory)
- **Architecture**: Two-layer LSTM with dropout
- **Use case**: Captures long-term dependencies in MT data
- **Parameters**: 128 hidden units (layer 1), 64 hidden units (layer 2)

### 2. GRU (Gated Recurrent Unit)
- **Architecture**: Two-layer GRU with dropout
- **Use case**: Simplified recurrent architecture, faster training
- **Parameters**: 128 hidden units (layer 1), 64 hidden units (layer 2)

### 3. Informer
- **Architecture**: Transformer-based model with ProbSparse attention
- **Use case**: Handles long sequences efficiently, captures complex patterns
- **Parameters**: 128 d_model, 4 attention heads, 2 encoder layers

## Workflow

### 1. Synthetic Data Generation
- Analyzes real MT data to extract statistical parameters
- Generates realistic resistivity models using geophysical constraints
- Creates corresponding MT responses using simplified forward modeling
- Produces 10,000 synthetic samples by default

### 2. Model Training
- Preprocesses data with standardization
- Trains models using Adam optimizer
- Validates performance on held-out data
- Saves trained models and scalers

### 3. Evaluation and Comparison
The project uses multiple metrics for comprehensive evaluation:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **Correlation**: Pearson correlation coefficient
- **SSIM**: Shape Similarity Index (custom metric)
- **CI**: Correlation Index (custom metric)

### 4. Real Data Prediction
- Loads trained models and scalers
- Processes EDI files to extract MT data
- Makes predictions on real data
- Compares with 1D inversion results

## Output Files

### Generated Files
- `{model_type}_model.pt`: Trained PyTorch model
- `{model_type}_scalers.pkl`: Data preprocessing scalers
- `{model_type}_training_curve.png`: Training/validation loss curves
- `{model_type}_validation_metrics.pkl/csv`: Performance metrics
- `{model_type}_validation_sample_{idx}.png`: Sample predictions
- `{site}_{model_type}_resistivity_comparison.png`: Real data predictions
- `synthetic_data.pkl`: Generated synthetic dataset
- `model_comparison.csv/png`: Cross-model performance comparison

## Key Features

### Reproducibility
- Fixed random seeds (default: 99)
- Consistent validation samples across models
- Deterministic CUDA operations

### Synthetic Data Quality
- Based on real MT data statistics
- Geophysically realistic resistivity models
- Proper skin depth considerations
- Noise simulation for robustness

### Model Validation
- Custom metrics for geophysical relevance
- Shape similarity assessment
- Statistical correlation analysis
- Visual comparison plots

## Example Results

The models typically achieve:
- **R² scores**: 0.85-0.95 on validation data
- **Correlation**: >0.9 between predicted and true profiles
- **RMSE**: <0.5 (on log-scaled resistivity)

## Customization

### Adding New Models
1. Create model class inheriting from `nn.Module`
2. Add to `create_model()` function
3. Update model type choices in argument parser

### Modifying Data Generation
- Adjust parameters in `generate_synthetic_data()`
- Modify resistivity distributions
- Change layer thickness calculations

### Custom Metrics
- Add new evaluation functions
- Update `calculate_and_save_validation_metrics()`
- Include in model comparison

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or model size
2. **File not found**: Check data directory structure
3. **Poor performance**: Increase training epochs or adjust hyperparameters
4. **Inconsistent results**: Ensure random seed is set consistently

### Performance Tips
- Use GPU for faster training
- Adjust batch size based on available memory
- Monitor training curves for overfitting
- Use early stopping if validation loss plateaus

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

<!-- ## License

[Add your license information here] -->

## Citation

If you use this code in your research, please cite:
```
@article{saibi2025comparison,
  title={Comparison of Deep Learning Models for 1D Magnetotelluric Inversion},
  author={SAIBI, Hakim and Hireche, Abdelhadi and Tsuji, Takeshi and Ali, Mohammed Y},
  year={2025}
}
```

## Contact

hakim.saibi@uaeu.ac.ae
