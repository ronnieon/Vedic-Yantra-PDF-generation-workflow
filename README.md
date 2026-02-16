# AI-Powered Storybook Generator 📚

A multi-phase Streamlit application that generates illustrated storybooks using AI models via Replicate.

## Features

- **5-Phase Workflow**: Structured pipeline from story conception to final PDF
- **AI-Generated Illustrations**: Uses Qwen Image for high-quality image generation
- **Simple Draft Sketches**: Creates clean black & white line drawings for composition planning
- **Draft-to-Final Pipeline**: Transforms minimal sketches into detailed, full-color illustrations
- **Asset Management**: Design consistent characters, artifacts, and environments
- **PDF Export**: Professional storybook format with alternating images and text

## Installation

1. **Clone or navigate to the project directory**

2. **Run the setup script (recommended):**
   ```bash
   ./setup.sh
   ```
   This will:
   - Create a Python virtual environment (`.venv`)
   - Create your `.envrc` file from the template
   - Open it for you to add your API keys
   - Install all Python dependencies in the virtual environment

   **OR** do it manually:

3. **Manual setup:**
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate it
   source .venv/bin/activate
   
   # Create environment file
   cp .envrc.example .envrc
   
   # Edit and add your API keys
   nano .envrc
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Install dependencies (if not using setup script):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API keys:**
   - Copy `.envrc.example` to `.envrc` (if not already created)
   - Add your Replicate API token to `.envrc`:
   ```bash
   export REPLICATE_API_TOKEN=your_token_here
   ```
   - Or enter it directly in the app sidebar

## Running the Application

**Important: Set up your API keys first!**

```bash
# Quick start (automatically activates venv and loads environment)
./run.sh
```

**Or manually:**

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Load environment variables
source .envrc

# 3. Start the application
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

**Note:** The `./run.sh` script automatically:
- Activates the virtual environment
- Loads your API keys from `.envrc`
- Starts the Streamlit application

## Usage Guide

### Phase 1: Story Setup 📝
1. Input your story title
2. Define the number of scenes and describe each scene
3. Add characters, artifacts, and environments that will appear in your story

### Phase 2: Draft Sketches ✏️
1. Click "Generate Draft Sketches"
2. The AI creates simple black & white line drawings with stick figures
3. Each character has a name tag with their current emotion (e.g., "LUNA [CURIOUS]")
4. Backgrounds are indicated with text boxes (e.g., "FOREST", "NIGHT SKY")
5. These sketches establish the basic composition and character positioning
6. Review the layout of each scene

### Phase 3: Asset Design 🎨
1. Click "Design Assets"
2. The AI generates detailed reference images for each character, artifact, and environment
3. These assets will inform the visual style of the final scenes

### Phase 4: Final Rendering ✨
1. Click "Render Final Scenes"
2. The AI transforms the simple sketches into full-color, detailed illustrations
3. Stick figures are replaced with actual characters based on name tags
4. Characters display the emotions indicated in their labels
5. Text boxes are replaced with actual detailed backgrounds
6. The scene description and asset designs guide the visual details
7. The composition from the sketch is used as the layout foundation

### Phase 5: PDF Export 📚
1. Click "Generate PDF Storybook"
2. The app compiles a PDF with:
   - Title page
   - Alternating image and text pages for each scene
3. Download your complete storybook!

## Architecture

### Key Components

- **Session State Management**: All data is preserved across reruns using `st.session_state`
- **Modular Functions**: Separate functions for image generation, editing, and PDF creation
- **Error Handling**: Comprehensive try-catch blocks and user-friendly error messages
- **Progress Indicators**: `st.spinner` and progress bars for long operations

### AI Models Used

- **Image Generation**: `qwen/qwen-image` (Qwen AI image generation)
- **Image Editing**: `qwen/qwen-image-edit-plus` (Qwen AI image refinement and editing)

### PDF Structure

```
Page 1: Title Page (Story Title)
Page 2: Image - Scene 1
Page 3: Text - Scene 1
Page 4: Image - Scene 2
Page 5: Text - Scene 2
...
```

## Technical Details

### Dependencies
- **streamlit**: Web application framework
- **replicate**: Python client for Replicate API
- **fpdf**: PDF generation library
- **Pillow**: Image processing
- **requests**: HTTP library for downloading images

### Constants Configuration

Model versions are defined at the top of `app.py` for easy updates:

```python
QWEN_IMAGE_MODEL = "tencent-arc/qwen-image:latest"
QWEN_IMAGE_EDIT_MODEL = "tencent-arc/qwen-image-edit-plus:latest"
```

### Session State Variables

- `story_title`: Story title string
- `scenes`: List of scene descriptions
- `characters`, `artifacts`, `environments`: Asset lists
- `draft_images`: Stick figure sketches
- `asset_designs`: Detailed asset references
- `final_images`: Refined final illustrations
- `api_token`: Replicate API token

## Troubleshooting

### API Errors
- Ensure your Replicate API token is valid and has sufficient credits
- Check the Replicate status page for service disruptions

### Image Generation Failures
- The app includes automatic retry logic and error handling
- Failed images are logged but don't stop the entire process

### PDF Generation Issues
- Ensure you have write permissions to the `/tmp` directory
- Large images may take longer to process

## Best Practices

1. **Start Small**: Begin with 2-3 scenes to test the workflow
2. **Be Descriptive**: Detailed prompts yield better results
3. **Review Drafts**: Check stick figure layouts before proceeding to final rendering
4. **Consistent Assets**: Define all characters and environments upfront for visual consistency

## Rate Limiting

The application includes 1-second delays between API calls to respect rate limits. You can adjust this in the code if needed.

## License

This is a demonstration application. Please ensure compliance with Replicate's terms of service.

## Support

For issues related to:
- **Streamlit**: Check the [Streamlit documentation](https://docs.streamlit.io/)
- **Replicate API**: Visit [Replicate's documentation](https://replicate.com/docs)
- **AI Models**: Refer to the Qwen model pages on Replicate

## Security

⚠️ **Important**: Your API keys are sensitive. Please read [SECURITY.md](SECURITY.md) for best practices on protecting your credentials.

---

**Created with ❤️ using Streamlit & Qwen AI Models**
