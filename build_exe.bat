@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r requirements.txt pyinstaller

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

"%PYTHON_EXE%" -m PyInstaller --noconfirm --onefile --windowed --name AudioSplitRouter ^
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
