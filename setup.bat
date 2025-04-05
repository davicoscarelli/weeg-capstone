@echo off

REM Create virtual environment
python -m venv weeg-env

REM Activate virtual environment
call weeg-env\Scripts\activate.bat

REM Install required packages
pip install -r requirements.txt

REM Register the virtual environment with Jupyter
python -m ipykernel install --user --name=weeg-env --display-name="Python (weeg-env)"

echo Setup complete! You can now run 'jupyter notebook' to start working with the notebook.
echo Make sure to select the 'Python (weeg-env)' kernel from the Kernel menu.

pause 