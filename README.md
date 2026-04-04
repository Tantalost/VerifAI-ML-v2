# VerifAI-ML: AI-Generated Image Detection System

A comprehensive machine learning system designed to detect AI-generated images with high accuracy and provide explainable AI insights through advanced visualization techniques.

## 🎯 Project Overview

VerifAI-ML leverages state-of-the-art YOLOv8 classification models combined with cutting-edge explainability techniques to distinguish between real and AI-generated images. The system provides not just predictions, but visual evidence highlighting the specific regions that indicate AI generation.

### Key Features
- **High Accuracy**: Target 80-90% classification accuracy
- **Explainable AI**: EigenCAM and Grad-CAM visualizations showing decision evidence
- **Batch Processing**: Handle single images or multiple uploads
- **Modern UI**: Streamlit-based interface with interactive visualizations
- **Performance Metrics**: Comprehensive evaluation and confidence scoring

## 📊 Dataset

- **Total Images**: 10,000 images
- **Classes**: 
  - 5,000 Real Images
  - 5,000 AI-Generated Images
- **Formats**: Standard image formats (.jpg, .png, .bmp)
- **Structure**: Organized in `dataset/real/` and `dataset/ai_generated/` directories

## 🏗️ System Architecture

The project is built in 4 phases, each building upon the previous:

### Phase 1: Data Preparation & Augmentation
**File**: `data_preparation.py`

**Purpose**: Prepare and augment the dataset for optimal training performance.

**Key Features**:
- **YOLOv8 Structure**: Creates required directory structure (`train/val/test` with class subdirectories)
- **Aggressive Augmentation**: 9 different techniques to prevent overfitting:
  - Horizontal flips (70% probability)
  - Random rotations (-15° to +15°)
  - Brightness/contrast adjustments
  - Gaussian blur
  - Noise injection
  - Saturation changes
  - Crop and resize operations

**Data Split**:
- Train: 80% (with heavy augmentation)
- Validation: 10% (no augmentation)
- Test: 10% (no augmentation)

**Output**: `yolo_dataset/` directory with properly structured data

### Phase 2: Training Pipeline
**File**: `training_pipeline.py`

**Purpose**: Train a high-performance YOLOv8 classification model using transfer learning.

**Key Features**:
- **Transfer Learning**: Starts with pre-trained `yolov8m-cls.pt` model
- **Smart Batch Sizing**: Automatically adjusts based on GPU memory
- **Early Stopping**: 15-epoch patience to prevent overfitting
- **Advanced Augmentation**: Built-in YOLOv8 augmentations plus custom ones
- **Mixed Precision**: AMP training for faster performance

**Training Parameters**:
- **Epochs**: 100 with early stopping
- **Learning Rate**: 0.001 with cosine annealing
- **Optimizer**: AdamW with weight decay
- **Image Size**: 224x224 (standard for classification)
- **Validation**: Every epoch with comprehensive metrics

**Evaluation Metrics**:
- Top-1 and Top-5 accuracy
- Precision, Recall, F1-Score
- Training curves visualization
- Test set performance analysis

**Output**: Trained model (`best.pt`) and training artifacts

### Phase 3: Explainability Module
**File**: `explainability_module.py`

**Purpose**: Generate visual explanations showing which image regions influence the AI/Real classification.

**Key Features**:
- **EigenCAM**: Principal Component Analysis on feature maps
- **Grad-CAM**: Gradient-based class activation mapping
- **Automatic Layer Detection**: Finds optimal convolutional layers
- **Batch Processing**: Handle multiple images efficiently

**Visualization Outputs**:
1. **Original Image**: Input image
2. **Heatmap**: CAM showing influential regions
3. **Overlay**: Heatmap superimposed on original image
4. **Prediction Details**: Class labels with confidence scores

**Technical Implementation**:
- **Feature Map Extraction**: Captures activations from backbone and neck layers
- **Principal Component Analysis**: EigenCAM using eigenvectors of feature covariance
- **Gradient Computation**: Grad-CAM with proper gradient flow
- **Smart Resizing**: Maintains spatial relationships in heatmaps

**Usage Example**:
```python
from explainability_module import YOLOv8Explainer

explainer = YOLOv8Explainer('path/to/trained/model.pt')
result = explainer.explain_image('image.jpg', method='eigencam')
```

### Phase 4: Streamlit UI
**File**: `app.py` (Coming Next)

**Purpose**: User-friendly interface for image classification and visualization.

**Planned Features**:
- **Sidebar Upload**: Single images or batch uploads (zip/multiple files)
- **Main Dashboard**: Image processing and results display
- **Per-Image Display**:
  - Original image
  - Heatmap overlay (evidence)
  - Predicted class
  - Confidence score bar chart
- **Batch Analytics**: Aggregate graphs showing AI vs Real distribution

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd VerifAI-ML
```

2. **Install dependencies by phase**:

**Phase 1 (Data Preparation)**:
```bash
pip install -r requirements_phase1.txt
```

**Phase 2 (Training)**:
```bash
pip install -r requirements_phase2.txt
```

**Phase 3 (Explainability)**:
```bash
pip install -r requirements_phase3.txt
```

**All Dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Data Preparation
```bash
python data_preparation.py
```
This will:
- Read images from `dataset/real/` and `dataset/ai_generated/`
- Create `yolo_dataset/` with proper YOLOv8 structure
- Apply aggressive augmentation to training data
- Generate dataset summary

#### Step 2: Model Training
```bash
python training_pipeline.py
```
This will:
- Load pre-trained YOLOv8 classification model
- Train with optimal parameters and early stopping
- Evaluate on test set with detailed metrics
- Generate training curves and performance plots
- Save the best model as `best.pt`

#### Step 3: Test Explainability
```python
from explainability_module import YOLOv8Explainer

# Initialize with your trained model
explainer = YOLOv8Explainer('path/to/best.pt')

# Explain a single image
result = explainer.explain_image('test_image.jpg', method='eigencam')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch processing
results = explainer.batch_explain(['img1.jpg', 'img2.jpg'], method='gradcam')
```

#### Step 4: Launch UI (Coming Soon)
```bash
streamlit run app.py
```

## 📁 Project Structure

```
VerifAI-ML/
├── dataset/                          # Raw dataset
│   ├── real/                         # Real images
│   └── ai_generated/                 # AI-generated images
├── yolo_dataset/                     # Processed dataset (created by Phase 1)
│   ├── train/
│   ├── val/
│   └── test/
├── data_preparation.py               # Phase 1: Data preparation script
├── training_pipeline.py              # Phase 2: Training script
├── explainability_module.py           # Phase 3: Explainability module
├── app.py                           # Phase 4: Streamlit UI (coming)
├── requirements.txt                 # All dependencies
├── requirements_phase1.txt          # Phase 1 dependencies
├── requirements_phase2.txt          # Phase 2 dependencies
├── requirements_phase3.txt          # Phase 3 dependencies
├── README.md                        # This file
└── runs/                            # Training outputs (created by Phase 2)
```

## 🎯 Performance Targets

### Accuracy Goals
- **Primary Target**: 80-90% classification accuracy
- **Secondary Metrics**: High precision and recall for both classes
- **Robustness**: Consistent performance across different image types

### Explainability Goals
- **Visual Clarity**: Clear heatmaps showing decision regions
- **Interpretability**: Understandable evidence for AI vs Real classification
- **Reliability**: Consistent explanations for similar image types

## 🔧 Technical Details

### Model Architecture
- **Base Model**: YOLOv8 Classification (yolov8m-cls.pt)
- **Input Size**: 224x224 RGB images
- **Output Classes**: 2 (Real, AI-Generated)
- **Parameters**: ~25M (medium-sized model for balance of speed and accuracy)

### Data Augmentation Strategy
- **Geometric**: Flips, rotations, crops, scaling
- **Photometric**: Brightness, contrast, saturation adjustments
- **Noise**: Gaussian blur, random noise injection
- **Advanced**: Mixup, CutMix (built into YOLOv8)

### Explainability Techniques

#### EigenCAM
- Uses Principal Component Analysis on feature maps
- Captures the most important patterns across multiple layers
- Good for showing overall decision patterns

#### Grad-CAM
- Uses gradient information to focus on class-specific features
- Provides fine-grained localization of important regions
- Excellent for pinpointing specific AI artifacts

## 📈 Expected Results

### Classification Performance
Based on similar AI detection research:
- **Accuracy**: 85-92% (with proper data augmentation)
- **Precision**: 80-90% (minimizing false positives)
- **Recall**: 85-95% (catching most AI-generated content)
- **F1-Score**: 82-92% (balanced performance)

### Explainability Insights
- **AI Artifacts**: Heatmaps will highlight:
  - Unnatural textures
  - Inconsistent lighting
  - Strange object boundaries
  - Repetitive patterns

- **Real Image Features**: Heatmaps will show:
  - Natural lighting variations
  - Consistent textures
  - Realistic object boundaries
  - Natural imperfections

## 🚨 Limitations & Considerations

### Current Limitations
- **Training Data Quality**: Performance depends on dataset quality and diversity
- **AI Model Evolution**: New AI generation methods may require model updates
- **Computational Resources**: Training requires significant GPU memory
- **Edge Cases**: Some images may be ambiguous even for humans

### Future Improvements
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Real-time Processing**: Optimize for faster inference
- **Mobile Deployment**: Create lightweight versions for mobile apps
- **Continuous Learning**: Update model with new AI generation techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLOv8 framework
- **PyTorch**: For deep learning infrastructure
- **Streamlit**: For the UI framework
- **Research Community**: For advancing AI detection techniques

---

## 📞 Contact

For questions, suggestions, or collaborations, please reach out through the project repository.

---

**Current Status**: Phase 3 Complete ✅ | Phase 4 (UI) In Progress 🚧

**Next Steps**: Complete Streamlit UI development and integration testing.
