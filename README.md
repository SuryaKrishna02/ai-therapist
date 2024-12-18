# AI Therapist

The repository consists of the code for the dataset preparation and model training of the AI therapist.

## Installation Guide

### 1. Dependencies Installation

#### On Linux
Run the following commands to update the package list and install the required system dependencies:
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6 -y
```

#### On Windows
You have two options to install FFmpeg:

Option 1 - Using Chocolatey:
1. Install [Chocolatey](https://chocolatey.org/install) (if not already installed)
2. Run in Administrator Command Prompt or PowerShell:
```powershell
choco install ffmpeg
```

Option 2 - Using Windows Package Manager:
```bash
winget install ffmpeg
```

Note: The libraries `libsm6` and `libxext6` are typically not required on Windows. If your application specifically needs equivalent functionality, consult your library's documentation for Windows-compatible alternatives.

### 2. Python Dependencies
Install all required Python packages by running:
```bash
pip install -r requirements.txt
```