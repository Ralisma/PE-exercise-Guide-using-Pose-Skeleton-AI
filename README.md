# Portable Physical Education Guide System

This is a portable version of the PE Guide system that can be run on Windows PCs with a camera.

## Setup Instructions

1. Ensure you have Python 3.7+ installed on your system.
2. The virtual environment and dependencies are already set up in the `venv` folder.
3. Place your Teachable Machine model file as `my_model/model.tflite`.

## Running the System

Double-click `run_pe_guide.bat` to start the system.

Alternatively, from command line:
```
cd pe_guide_portable
venv\Scripts\activate
python pe_guide.py
```

## Files Structure

- `pe_guide.py`: Main Python script
- `my_model/`: Directory for your TensorFlow Lite model
- `logs/`: Directory for log files
- `exercise_log.csv`: CSV file for exercise logs
- `venv/`: Virtual environment with dependencies
- `run_pe_guide.bat`: Batch file to run the system

## Requirements

- Windows PC with camera
- Python 3.7+
- TensorFlow Lite model file

## Notes

- The system uses pyttsx3 for text-to-speech (works offline)
- Beeps are generated using SoX (if available) or fallback to system sounds
- Logs are saved to `exercise_log.csv`
