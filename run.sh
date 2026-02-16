#!/bin/bash

# AI Storybook Generator - Startup Script

echo "🚀 Starting AI Storybook Generator..."
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Load environment variables from .envrc
if [ -f .envrc ]; then
    echo "📋 Loading environment variables from .envrc..."
    source .envrc
else
    echo "⚠️  Warning: .envrc file not found. Please create it with your API keys."
fi

echo "✅ Environment configured"
echo "🌐 Launching Streamlit app..."
echo ""

# Run the Streamlit app
streamlit run app.py
