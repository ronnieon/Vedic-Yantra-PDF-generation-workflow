#!/bin/bash

# Setup Script for AI Storybook Generator

echo "🔧 AI Storybook Generator - Initial Setup"
echo "=========================================="
echo ""

# Check if .envrc already exists
if [ -f .envrc ]; then
    echo "✅ .envrc file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⏭️  Skipping .envrc creation"
    else
        cp .envrc.example .envrc
        echo "📝 Created new .envrc from template"
    fi
else
    cp .envrc.example .envrc
    echo "📝 Created .envrc from template"
fi

echo ""
echo "⚠️  IMPORTANT: Edit .envrc and add your API keys!"
echo "   Required: REPLICATE_API_TOKEN"
echo ""
read -p "Press Enter to open .envrc in your default editor..."
${EDITOR:-nano} .envrc

echo ""
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "📦 Installing Python dependencies in virtual environment..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source .envrc"
echo "  streamlit run app.py"
echo ""
