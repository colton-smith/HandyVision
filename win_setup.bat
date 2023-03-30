@echo off
REM Batch script to setup HandyVision python requirements

@echo ---------------------------------------------------
@echo Creating python virtual environment
@echo ---------------------------------------------------
python3 -m venv .venv

@echo ---------------------------------------------------
@echo Activating python virtual environment
@echo ---------------------------------------------------
start /min /wait "" cmd /c .\.venv\Scripts\activate

@echo ---------------------------------------------------
@echo Installing python requirements
@echo ---------------------------------------------------
pip install -r requirements.txt

@echo ---------------------------------------------------
@echo Installing HandyVision package in development mode
@echo ---------------------------------------------------
pip install -e lib
