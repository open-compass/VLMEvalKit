# MDPBench Evaluation Pipeline

MDPBench is a specialized dataset for Multimodal Document Processing and OCR evaluation within the VLMEvalKit framework.

## 🚀 Installation & Environment Setup

### 1. Install Python Dependencies
All required Python packages, including standard metrics and CDM (Comprehensive Distance Metric) evaluation dependencies, are unified in `vlmeval/dataset/MDPBench/requirements.txt`:

From the VLMEvalKit repository root:

```bash
pip install -r vlmeval/dataset/MDPBench/requirements.txt
```

### 2. Configure CDM Environment
CDM metric performs visual rendering comparison for complex formulas and tables. It dynamically detects whether the system environment is correctly configured. If missing, it will gracefully degrade and skip the CDM score without crashing.

To **enable CDM**, install the following system-level packages:

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y nodejs npm imagemagick texlive texlive-latex-extra
```

**macOS:**
```bash
brew install node imagemagick texlive
```
