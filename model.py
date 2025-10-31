import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
from typing import Optional, Union, Dict, Any
import os


class HybridModel(nn.Module):
    """
    Hybrid Model for TrustLens - A flexible architecture that can handle
    various input types and provide trust analysis capabilities.
    """
    
    def __init__(self, input_size: int = 224, num_classes: int = 2, hidden_dim: int = 512):
        super(HybridModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Calculate the flattened size
        self.flattened_size = 512 * 7 * 7
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Trust analysis layers
        self.trust_analyzer = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        
        # Classification output
        classification = self.classifier(features_flat)
        
        # Trust score
        trust_score = self.trust_analyzer(features_flat)
        
        return {
            'classification': classification,
            'trust_score': trust_score,
            'features': features_flat
        }


class ModelManager:
    """
    Manages loading and inference for different model formats (PyTorch, ONNX)
    """
    
    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir
        self.pytorch_model = None
        self.onnx_session = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pytorch_model(self, weights_path: str, model_config: Optional[Dict] = None) -> bool:
        """Load PyTorch model from weights file"""
        try:
            if model_config is None:
                model_config = {"input_size": 32, "num_classes": 2, "hidden_dim": 512}  # Use 32x32 input
            
            self.pytorch_model = HybridModel(**model_config)
            
            # Load weights
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.pytorch_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.pytorch_model.load_state_dict(checkpoint)
                else:
                    self.pytorch_model.load_state_dict(checkpoint)
                
                self.pytorch_model.to(self.device)
                self.pytorch_model.eval()
                return True
            else:
                print(f"Weights file not found: {weights_path}")
                return False
                
        except Exception as e:
            print(f"Error loading PyTorch model: {str(e)}")
            return False
    
    def load_onnx_model(self, onnx_path: str) -> bool:
        """Load ONNX model"""
        try:
            if os.path.exists(onnx_path):
                providers = ['CPUExecutionProvider']
                if torch.cuda.is_available():
                    providers.insert(0, 'CUDAExecutionProvider')
                
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                return True
            else:
                print(f"ONNX file not found: {onnx_path}")
                return False
                
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")
            return False
    
    def predict_pytorch(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run inference with PyTorch model"""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model not loaded")
        
        try:
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                outputs = self.pytorch_model(input_tensor)
                
                # Apply softmax to classification output
                classification_probs = F.softmax(outputs['classification'], dim=1)
                predicted_class = torch.argmax(classification_probs, dim=1)
                
                return {
                    'classification_probs': classification_probs.cpu().numpy(),
                    'predicted_class': predicted_class.cpu().numpy(),
                    'trust_score': outputs['trust_score'].cpu().numpy(),
                    'features': outputs['features'].cpu().numpy()
                }
        except Exception as e:
            raise RuntimeError(f"PyTorch inference failed: {str(e)}")
    
    def predict_onnx(self, input_array: np.ndarray) -> Dict[str, Any]:
        """Run inference with ONNX model"""
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded")
        
        try:
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_array})
            
            # Assuming the ONNX model outputs classification logits
            classification_probs = F.softmax(torch.tensor(outputs[0]), dim=1).numpy()
            predicted_class = np.argmax(classification_probs, axis=1)
            
            return {
                'classification_probs': classification_probs,
                'predicted_class': predicted_class,
                'raw_outputs': outputs
            }
            
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'pytorch_loaded': self.pytorch_model is not None,
            'onnx_loaded': self.onnx_session is not None,
            'device': str(self.device)
        }
        
        # Add ONNX model input/output info if available
        if self.onnx_session is not None:
            inputs = self.onnx_session.get_inputs()
            outputs = self.onnx_session.get_outputs()
            
            info['onnx_inputs'] = [
                {
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                } for inp in inputs
            ]
            
            info['onnx_outputs'] = [
                {
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                } for out in outputs
            ]
        
        return info
    
    def get_onnx_input_size(self) -> tuple:
        """Get the expected input size for ONNX model"""
        if self.onnx_session is None:
            return (224, 224)  # Default size
        
        inputs = self.onnx_session.get_inputs()
        if inputs:
            # Assuming input shape is [batch, channels, height, width]
            shape = inputs[0].shape
            if len(shape) >= 4:
                height, width = shape[2], shape[3]
                return (height, width)
        
        return (224, 224)  # Default fallback


def preprocess_image(image: np.ndarray, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Preprocess image for model inference - matches training preprocessing
    """
    import cv2
    
    # Resize image
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    if len(image.shape) == 3:
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
    
    tensor = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    
    # Apply the same normalization as training: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # This transforms [0, 1] to [-1, 1] range
    tensor = (tensor - 0.5) / 0.5
    
    return tensor