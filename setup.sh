#!/bin/bash

# Create virtual environment
python3 -m venv weeg-env

# Activate virtual environment
source weeg-env/bin/activate

# Install required packages
pip install -r requirements.txt

# Register the virtual environment with Jupyter
python -m ipykernel install --user --name=weeg-env --display-name="Python (weeg-env)"

echo "Setup complete! You can now run 'jupyter notebook' to start working with the notebook."
echo "Make sure to select the 'Python (weeg-env)' kernel from the Kernel menu." 