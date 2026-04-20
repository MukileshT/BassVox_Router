@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --onefile --windowed --name AudioSplitRouter ^
    --hidden-import=numpy ^
    --hidden-import=scipy ^
    --hidden-import=scipy.signal ^
    --hidden-import=sounddevice ^
    --hidden-import=soundfile ^
    --hidden-import=PySide6 ^
    --collect-all=PySide6 ^
    app.py

echo.
echo Build complete. EXE is in: dist\AudioSplitRouter.exe
pause
