# OmniReg Hybrid OCR System

## Overview

OmniReg is a desktop OCR system for mathematical expressions with a responsive Tkinter interface.
It combines:

- OpenCV adaptive preprocessing for clean binary equation images
- Optional Gemini transcription for online formula extraction
- Local PyTorch + JiWER metric simulation for fast, consistent evaluation output

The app supports an offline demo mode when Gemini is not configured.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/omnireg.git
cd omnireg
```

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional online mode setup (Gemini)

Set one of these environment variables before launching:

- GEMINI_API_KEY
- GOOGLE_API_KEY

If no key is set, OmniReg runs in Offline Demo Mode.

## Running the Project

### Standard launcher

```bash
python launch.py
```

### Direct GUI launch

```bash
python gui_folder/main_gui.py
```

## Dataset

Place evaluation images in the dataset folder:

- dataset/

Supported input formats include PNG, JPG, JPEG, BMP, and WEBP.

## Testing

Current lightweight validation:

```bash
python -c "from gui_folder.main_gui import OmniRegGUI; print('gui-import-ok')"
```

You can also run the app and verify:

- image browse and preview rendering
- adaptive threshold output in the binary panel
- metric updates in the performance board

## Architecture

- launch.py: Cross-platform startup and environment checks
- gui_folder/main_gui.py: Main Tkinter application and inference thread orchestration
- dataset/: Local input images
- hf_cache/: Local model/cache directory used by runtime configuration

Runtime flow:

1. User selects an image in the UI
2. OpenCV adaptive threshold preprocessing generates a crisp binary view
3. Gemini transcription runs when configured, otherwise offline fallback is used
4. PyTorch and JiWER metrics are computed and displayed in the Model Performance board


## Contributors

- Group 5
- Ronald Cabral
- Jersey Estrella
- Evan Landong