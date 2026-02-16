#!/bin/bash

echo "🔧 Setting up AI Storybook Generator..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 is required but not found!"
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.12 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for .envrc
if [ ! -f ".envrc" ]; then
    echo "⚠️ .envrc file not found!"
    echo "Please create .envrc with your API keys:"
    echo "  export REPLICATE_API_TOKEN=your_token"
    echo "  export GEMINI_API_KEY=your_key"
    echo "  export ELEVENLABS_API_KEY=your_key"
else
    echo "✅ .envrc file found"
fi

echo ""
echo "✅ Setup complete!"
echo "To run the app: ./run.sh"
