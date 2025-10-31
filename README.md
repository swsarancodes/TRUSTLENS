# 🔍 TrustLens - AI Model Trust and Transparency Analysis

TrustLens is a comprehensive Streamlit application for analyzing AI model trust, transparency, and performance. It supports both PyTorch and ONNX models and provides detailed insights into model predictions and confidence scores.

## 🚀 Features

- **Multi-Model Support**: Load and compare PyTorch (.pth) and ONNX (.onnx) models
- **Interactive Inference**: Upload images or use sample data for real-time predictions
- **Trust Analysis**: Advanced trust scoring and confidence analysis
- **Comprehensive Analytics**: Detailed performance metrics and visualizations
- **Model Comparison**: Side-by-side comparison of different model types
- **Export Capabilities**: Export inference history as CSV or JSON
- **Modern UI**: Beautiful, responsive interface with real-time updates

## 📁 Project Structure

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

## 🛠️ Installation

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

## 🚀 Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Load Models**:
   - Click "🔄 Load Models" in the sidebar
   - The app will automatically detect and load available model files

3. **Run Inference**:
   - Choose input method (Upload Image, Sample Data, or Random Input)
   - Select model type (PyTorch or ONNX)
   - Click "🚀 Run Inference"

4. **Analyze Results**:
   - View predictions, confidence scores, and trust metrics
   - Explore detailed analytics and visualizations
   - Compare different models and export results

## 📊 Model Architecture

The hybrid model includes:

- **Feature Extractor**: Convolutional layers for image feature extraction
- **Classifier**: Multi-layer perceptron for classification
- **Trust Analyzer**: Specialized layers for trust score computation

### Model Configuration

```python
model_config = {
    "input_size": 224,      # Input image size
    "num_classes": 2,       # Number of output classes
    "hidden_dim": 512       # Hidden layer dimension
}
```

## 🔧 Configuration

### Model Settings

- **Confidence Threshold**: Minimum confidence for high-confidence predictions
- **Trust Threshold**: Minimum trust score for reliable predictions
- **Model Type**: Choose between PyTorch and ONNX inference

### Input Formats

- **Images**: PNG, JPG, JPEG, BMP, TIFF
- **Size**: Automatically resized to 224x224
- **Channels**: RGB (3 channels)

## 📈 Analytics Features

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

## 🎯 Model Training Integration

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

## 🔍 Trust Analysis

TrustLens provides advanced trust metrics:

- **Confidence Score**: Model's certainty in predictions
- **Trust Score**: Reliability assessment based on feature analysis
- **Feature Activation**: Visualization of important features
- **Prediction Stability**: Consistency across similar inputs

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure model files are in the correct directory
   - Check file permissions and integrity
   - Verify model architecture compatibility

2. **CUDA Issues**:
   - The app automatically detects GPU availability
   - Falls back to CPU if CUDA is unavailable

3. **Memory Issues**:
   - Reduce batch size for large images
   - Close other applications to free memory

4. **Import Errors**:
   - Ensure all dependencies are installed
   - Check Python version compatibility (>=3.8)

### Performance Tips

- Use ONNX models for faster inference
- Enable GPU acceleration when available
- Optimize image sizes before upload
- Clear inference history periodically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the interactive interface
- PyTorch for deep learning capabilities
- ONNX for model optimization
- Plotly for advanced visualizations

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Analyzing! 🔍✨**