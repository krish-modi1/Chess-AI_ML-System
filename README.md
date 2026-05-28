# EE542 Chess AI - PocketZero

## Submitted by: Krish Modi | Mohith Suresh | Adithya Srivastava

## 1. Download the Model

1. Visit: https://drive.google.com/file/d/1FHQQI9hNmIxAZd6zmX6QO8oow5ekjgGs/view?usp=sharing
2. Click the download button
3. Place the downloaded model file in `game_engine/model/best_model.pth`

## 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Run the Application

```bash
# Start playing against the model
python app.py
```

The app will prompt you to open a browser. Visit the local URL (usually `http://localhost:5000`) to play against the model.

---

## Verification Checklist

Before running, ensure:
- ✓ `game_engine/model/best_model.pth` exists
- ✓ `requirements.txt` exists in project root
- ✓ Virtual environment is activated
- ✓ All dependencies installed: `pip list`
- ✓ `app.py` exists and is executable
