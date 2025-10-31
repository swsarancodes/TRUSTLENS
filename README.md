

# TrustLens - AI Model Trust and Transparency Analysis

https://github.com/user-attachments/assets/901056d5-0355-499f-b6e5-4a342c6759c4

TrustLens is a comprehensive Streamlit application for analyzing AI model trust, transparency, and performance. It supportsONNX models and provides detailed insights into model predictions and confidence scores. This implementation focuses on a hybrid CNN-ViT model trained to distinguish between real and AI-generated images.

## Features

- Multi-Model Support: Load and compare PyTorch (.pth) and ONNX (.onnx) models
- Interactive Inference: Upload images or use sample data for real-time predictions
- Trust Analysis: Advanced trust scoring and confidence analysis
- Comprehensive Analytics: Detailed performance metrics and visualizations
- Model Comparison: Side-by-side comparison of different model types
- Export Capabilities: Export inference history as CSV or JSON
- Modern UI: Beautiful, responsive interface with real-time updates

## Project Structure

```
TRUSTLENS/
├── app.py                 # Main Streamlit application
├── model.py              # Model architecture and management
├── pyproject.toml        # UV package management configuration
├── requirements.txt      # Alternative dependency list
├── README.md            # This file
├── final_weights.pth    # Trained model weights
├── trustlens.pth        # Alternative model weights
└── hybrid_model.onnx    # ONNX optimized model
```

## Installation

### Option 1: Using UV (Recommended)

1. Install UV if you haven't already:
```bash
pip install uv
```

2. Install dependencies:
```bash
uv pip install -e .
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

### Option 3: Manual Installation

```bash
pip install torch torchvision streamlit numpy pandas matplotlib seaborn plotly pillow opencv-python scikit-learn tqdm onnx onnxruntime
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Load Models:
   - Click "Load Models" in the sidebar
   - The app will automatically detect and load available model files

3. Run Inference:
   - Choose input method (Upload Image, Sample Data, or Random Input)
   - Select model type (PyTorch or ONNX)
   - Click "Run Inference"

4. Analyze Results:
   - View predictions, confidence scores, and trust metrics
   - Explore detailed analytics and visualizations
   - Compare different models and export results

## Model Architecture

The hybrid model combines convolutional neural networks with vision transformers for robust image classification:

### CNN Block
- Three convolutional layers with batch normalization and ReLU activation
- Feature extraction from input images
- Output shape: 64 channels with 24x24 spatial dimensions

### Vision Transformer (ViT) Block
- Patch embedding with 5x5 patches
- 5 transformer encoder layers
- Multi-head self-attention with 128 heads
- 256-dimensional embedding space

### Attention Mechanism
- Custom attention mechanism for feature refinement
- Query-key-value attention with layer normalization
- 128-dimensional attention space

### Classifier
- Fully connected layers for final classification
- Dropout for regularization
- Binary classification output (REAL vs FAKE)

### Model Configuration

```python
model_config = {
    "input_size": 32,       # Input image size
    "num_classes": 2,       # Number of output classes (REAL, FAKE)
    "hidden_dim": 256,      # Hidden layer dimension
    "patch_size": 5,        # Size of patches for ViT
    "num_transformer_layers": 5,  # Number of transformer encoder layers
    "num_heads": 128,       # Number of attention heads
    "mlp_size": 2048        # Size of MLP in transformer
}
```

## Dataset

The model is trained on the CIFAKE dataset (Real and AI Generated Synthetic Images):
- Training set: 100,000 images (50,000 REAL, 50,000 FAKE)
- Test set: 20,000 images (10,000 REAL, 10,000 FAKE)
- Image size: 32x32 RGB images
- Classes: REAL (authentic photographs) and FAKE (AI-generated images)

### Data Preprocessing
- Random Gaussian blur (50% probability) for training data augmentation
- Resizing to 32x32 pixels
- Normalization with ImageNet statistics

## Training

The model was trained for 10 epochs with the following configuration:
- Optimizer: Adam (learning rate: 0.0001)
- Loss function: Cross-entropy loss
- Learning rate scheduler: ReduceLROnPlateau
- Batch size: 32
- Device: CUDA (GPU acceleration)

### Training Results
- Final training accuracy: 97.80%
- Final test accuracy: 94.11%
- Training loss decreased from 0.2024 to 0.0592
- Test loss decreased from 0.1806 to 0.1838

### Classification Performance
- Precision: 0.94 (both classes)
- Recall: 0.93 (FAKE), 0.95 (REAL)
- F1-score: 0.94 (both classes)

## Configuration

### Model Settings
- Confidence Threshold: Minimum confidence for high-confidence predictions
- Trust Threshold: Minimum trust score for reliable predictions
- Model Type: Choose between PyTorch and ONNX inference

### Input Formats
- Images: PNG, JPG, JPEG, BMP, TIFF
- Size: Automatically resized to 32x32
- Channels: RGB (3 channels)

## Analytics Features

### Real-time Metrics
- Prediction accuracy
- Confidence scores
- Trust analysis
- Feature visualization

### Historical Analysis
- Performance trends over time
- Class distribution analysis
- Confidence vs. trust correlation
- Model comparison metrics

### Export Options
- CSV format for spreadsheet analysis
- JSON format for programmatic access
- Visualization exports

## Model Training Integration

The application is designed to work with models trained using the provided boilerplate code structure:

```python
# Training loop integration
for epoch in range(epochs):
    # Training phase
    model.train()
    # ... training code ...
    
    # Evaluation phase
    model.eval()
    # ... evaluation code ...
    
    # Save checkpoints
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # ... other training state ...
    }, checkpoint_path)
```

## Trust Analysis

TrustLens provides advanced trust metrics:
- Confidence Score: Model's certainty in predictions
- Trust Score: Reliability assessment based on feature analysis
- Feature Activation: Visualization of important features
- Prediction Stability: Consistency across similar inputs

## ONNX Export

The model has been exported to ONNX format for optimized inference:
- Compatible with ONNX Runtime
- Dynamic batch size support
- Optimized for production deployment

## Troubleshooting

### Common Issues

1. Model Loading Errors:
   - Ensure model files are in the correct directory
   - Check file permissions and integrity
   - Verify model architecture compatibility

2. CUDA Issues:
   - The app automatically detects GPU availability
   - Falls back to CPU if CUDA is unavailable

3. Memory Issues:
   - Reduce batch size for large images
   - Close other applications to free memory

4. Import Errors:
   - Ensure all dependencies are installed
   - Check Python version compatibility (>=3.8)

### Performance Tips
- Use ONNX models for faster inference
- Enable GPU acceleration when available
- Optimize image sizes before upload
- Clear inference history periodically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit for the interactive interface
- PyTorch for deep learning capabilities
- ONNX for model optimization
- Plotly for advanced visualizations
- CIFAKE dataset for training and evaluation

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

Happy Analyzing!
