#!/bin/bash

# Make sure that the script is executed from the directory where main.py is located and requirements.txt
# Installing dependencies from requirements.txt
if [ -f requirements.txt ]; then
    echo "Вщцтдщфвштп dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "File requirements.txt is not found."
    exit 1
fi

# Launch main.py
if [ -f main.py ]; then
    echo "Launching main.py..."
    python main.py
else
    echo "File main.py is not found."
    exit 1
fi