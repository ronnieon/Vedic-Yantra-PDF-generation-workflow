#!/bin/bash

# Load environment variables
if [ -f .envrc ]; then
    source .envrc
    echo "✅ Environment variables loaded from .envrc"
else
    echo "⚠️ .envrc file not found!"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
streamlit run app.py
