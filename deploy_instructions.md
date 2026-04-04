# 🚀 VerifAI-ML Deployment Instructions

## Git Repository Size Issue Solution

The current repository has large files in its history (2.15 GiB), causing push failures. Here are two solutions:

### Option 1: Create New Clean Repository (Recommended)

1. **Create a new repository** on GitHub with a different name (e.g., `VerifAI-ML-v2`)
2. **Clone the new empty repository:**
   ```bash
   git clone https://github.com/yourusername/VerifAI-ML-v2.git
   cd VerifAI-ML-v2
   ```

3. **Copy only the essential files** from your current project:
   ```bash
   # Copy Python files only
   cp ../VerifAI-ML/*.py .
   cp ../VerifAI-ML/*.md .
   cp ../VerifAI-ML/*.txt .
   cp ../VerifAI-ML/.gitignore .
   ```

4. **Add and commit the clean files:**
   ```bash
   git add .
   git commit -m "Initial commit: VerifAI-ML system"
   git push origin main
   ```

5. **Add your dataset and train locally:**
   ```bash
   # Create dataset directory and add your images
   mkdir dataset
   # Add your images to dataset/real/ and dataset/ai_generated/
   
   # Run the complete setup
   python setup.py
   ```

### Option 2: Clean Current Repository (Advanced)

Use BFG Repo-Cleaner to remove large files from history:

1. **Download BFG Repo-Cleaner:**
   ```bash
   wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
   ```

2. **Remove large files from history:**
   ```bash
   java -jar bfg-1.14.0.jar --strip-biggest 100M .
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push origin main --force
   ```

## 🎯 Quick Start After Setup

Once you have a clean repository:

1. **Install dependencies:**
   ```bash
   pip install -r requirements_complete.txt
   ```

2. **Add your dataset:**
   ```
   VerifAI-ML/
   ├── dataset/
   │   ├── real/          # Add your real images here
   │   └── ai_generated/  # Add your AI images here
   ```

3. **Run the complete setup:**
   ```bash
   python setup.py
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## 📁 What's Included in the Clean Repo

- **Core Python Files:**
  - `data_preparation.py` - Data augmentation and splitting
  - `training_pipeline.py` - YOLOv8 training
  - `explainability_module.py` - AI explainability
  - `app.py` - Streamlit UI
  - `setup.py` - Automated setup
  - `run_app.py` - App launcher

- **Documentation:**
  - `README.md` - Complete project documentation
  - `requirements_complete.txt` - All dependencies
  - `.gitignore` - Proper exclusions for large files

- **Configuration:**
  - Proper `.gitignore` to exclude:
    - Dataset files (`dataset/`, `yolo_dataset/`)
    - Model files (`*.pt`, `*.pth`)
    - Training outputs (`runs/`)
    - Cache and temp files

## 🚨 Important Notes

- **Never commit** your dataset images to git
- **Never commit** trained model files (they're large)
- **Always use** `.gitignore` to exclude large files
- **Dataset and models** should be generated locally after cloning

## 🎉 Expected Repository Size

After cleanup, your repository should be:
- **< 50 MB** total size
- **Fast cloning** and pushing
- **Clean history** with only source code

This makes it easy for others to clone and use your system!
