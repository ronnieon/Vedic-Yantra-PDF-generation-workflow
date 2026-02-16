"""
AI-Powered Storybook PDF Generator
A multi-phase Streamlit application for creating illustrated storybooks using Qwen AI models.
"""

import streamlit as st
import replicate
from fpdf import FPDF
import base64
import io
import time
import requests
from PIL import Image
from typing import List, Dict, Optional
import os
import json
import google.genai as genai
from functools import wraps

# ============================================================================
# CONSTANTS - Model Versions
# ============================================================================
# Using Qwen for image generation
QWEN_IMAGE_MODEL = "qwen/qwen-image"
# Using Qwen for image editing
QWEN_IMAGE_EDIT_MODEL = "qwen/qwen-image-edit-plus"

# PDF Settings
PDF_WIDTH = 210  # A4 width in mm
PDF_HEIGHT = 297  # A4 height in mm
IMAGE_WIDTH = 180  # Image width in PDF
IMAGE_HEIGHT = 200  # Image height in PDF

# Color Palette for Asset Identification
# Distinct colors for color-coding characters and artifacts in sketches
COLOR_PALETTE = [
    "BRIGHT RED",
    "BRIGHT BLUE", 
    "BRIGHT GREEN",
    "BRIGHT YELLOW",
    "BRIGHT PURPLE",
    "BRIGHT ORANGE",
    "BRIGHT PINK",
    "BRIGHT CYAN",
    "BRIGHT MAGENTA",
    "BRIGHT LIME GREEN"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def retry_with_backoff(max_retries=3, initial_delay=2):
    """Decorator to retry API calls with exponential backoff for rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error
                    if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                        if attempt < max_retries - 1:
                            st.warning(f"⏳ Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                            continue
                        else:
                            st.error(
                                "❌ **Gemini API Rate Limit Exceeded**\n\n"
                                "Your Gemini API quota has been exhausted. Please:\n"
                                "1. Wait a few minutes and try again\n"
                                "2. Check your quota at: https://aistudio.google.com/app/apikey\n"
                                "3. Consider upgrading your API plan if needed\n"
                                "4. Reduce the number of scenes to process at once"
                            )
                    raise  # Re-raise the exception if not rate limit or max retries reached
            return None
        return wrapper
    return decorator

def initialize_session_state():
    """Initialize all session state variables."""
    if 'story_title' not in st.session_state:
        st.session_state.story_title = ""
    if 'scenes' not in st.session_state:
        st.session_state.scenes = [""]
    if 'characters' not in st.session_state:
        st.session_state.characters = [""]
    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = [""]
    if 'environments' not in st.session_state:
        st.session_state.environments = [""]
    if 'draft_images' not in st.session_state:
        st.session_state.draft_images = []
    if 'asset_designs' not in st.session_state:
        st.session_state.asset_designs = {}
    if 'final_images' not in st.session_state:
        st.session_state.final_images = []
    if 'api_token' not in st.session_state:
        # Try to load from environment variable
        st.session_state.api_token = os.environ.get('REPLICATE_API_TOKEN', "")
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = os.environ.get('GEMINI_API_KEY', "")
    if 'scenes_analyzed' not in st.session_state:
        st.session_state.scenes_analyzed = False
    if 'assets_confirmed' not in st.session_state:
        st.session_state.assets_confirmed = False
    if 'character_colors' not in st.session_state:
        st.session_state.character_colors = {}  # Maps character index to color
    if 'artifact_colors' not in st.session_state:
        st.session_state.artifact_colors = {}  # Maps artifact index to color


def download_image(url: str) -> Optional[bytes]:
    """Download image from URL and return as bytes."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Failed to download image: {str(e)}")
        return None


def analyze_scenes_with_gemini(scenes: List[str], api_key: str) -> Optional[Dict]:
    """
    Analyze scenes using Gemini to extract characters, artifacts, and environments.
    
    Args:
        scenes: List of scene descriptions
        api_key: Gemini API key
        
    Returns:
        Dictionary with characters, artifacts, and environments or None if failed
    """
    @retry_with_backoff(max_retries=3, initial_delay=3)
    def _analyze():
        client = genai.Client(api_key=api_key)
        
        scenes_text = "\n\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scenes) if scene.strip()])
        
        prompt = f"""Analyze the following story scenes and extract:
1. All unique CHARACTERS with their names and brief descriptions (appearance, personality traits)
2. All ARTIFACTS/OBJECTS that are important to the story (magical items, tools, etc.)
3. All ENVIRONMENTS/SETTINGS where scenes take place

Scenes:
{scenes_text}

Return your response in the following JSON format:
{{
  "characters": [
    "Character Name: Description with appearance and personality"
  ],
  "artifacts": [
    "Artifact Name: Description"
  ],
  "environments": [
    "Environment Name: Description"
  ]
}}

Be specific and include visual details that would help an illustrator."""

        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=prompt
        )
        
        # Extract JSON from response
        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")
            
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        return result
    
    try:
        return _analyze()
    except Exception as e:
        st.error(f"Failed to analyze scenes with Gemini: {str(e)}")
        return None


def generate_sketch_prompt_with_gemini(scene: str, characters: List[str], artifacts: List[str], 
                                        character_colors: Dict[int, str], artifact_colors: Dict[int, str],
                                        api_key: str) -> Optional[str]:
    """
    Use Gemini to generate a precise sketch prompt for a scene with color-coded assets.
    
    Args:
        scene: Scene description
        characters: List of character descriptions
        artifacts: List of artifact descriptions
        character_colors: Dict mapping character index to assigned color
        artifact_colors: Dict mapping artifact index to assigned color
        api_key: Gemini API key
        
    Returns:
        Detailed sketch prompt or None if failed
    """
    @retry_with_backoff(max_retries=3, initial_delay=3)
    def _generate():
        client = genai.Client(api_key=api_key)
        
        # Build character list with colors
        characters_with_colors = []
        for i, char in enumerate(characters):
            if char.strip():
                color = character_colors.get(i, "BLACK")
                characters_with_colors.append(f"- {char} [COLOR: {color}]")
        characters_list = "\n".join(characters_with_colors) if characters_with_colors else "No characters"
        
        # Build artifact list with colors
        artifacts_with_colors = []
        for i, artifact in enumerate(artifacts):
            if artifact.strip():
                color = artifact_colors.get(i, "BLACK")
                artifacts_with_colors.append(f"- {artifact} [COLOR: {color}]")
        artifacts_list = "\n".join(artifacts_with_colors) if artifacts_with_colors else "No artifacts"
        
        prompt = f"""You are a storybook illustrator creating a SIMPLE COLOR-CODED sketch. Analyze this scene and create a sketch prompt.

SCENE DESCRIPTION:
{scene}

AVAILABLE CHARACTERS (with assigned colors):
{characters_list}

AVAILABLE ARTIFACTS (with assigned colors):
{artifacts_list}

TASK: Create a simple sketch prompt following these rules:

START YOUR PROMPT WITH:
"Draw a VERY SIMPLE stick figure sketch. Use SOLID BRIGHT COLORS to fill each character and artifact as specified below. ABSOLUTELY NO TEXT, NO LABELS, NO NAME TAGS, NO TEXT BOXES. Clean white background."

THEN SPECIFY:

1. CHARACTER PLACEMENT:
   - For each character in this scene, write:
     * Position and pose (e.g., "Draw a stick figure standing on the left")
     * Emotion/gesture (e.g., "arms raised in joy", "kneeling down", "pointing forward")
     * COLOR: "Fill this entire stick figure with SOLID [COLOR]" using their exact assigned color
   - Example: "Draw a simple stick figure standing on the left side, arms outstretched. Fill this entire stick figure with SOLID BRIGHT RED."

2. ARTIFACT PLACEMENT (if any):
   - For each artifact in this scene, write:
     * Simple representation (e.g., "Draw a simple wand shape", "Draw a circular crown")
     * Position (e.g., "held in the right figure's hand", "floating above the center")
     * COLOR: "Fill this object entirely with SOLID [COLOR]" using its exact assigned color
   - Example: "Draw a simple stick/wand shape in the figure's hand. Fill this wand entirely with SOLID BRIGHT BLUE."

3. BACKGROUND:
   - Keep background MINIMAL - just a few simple lines if needed (horizon line, simple ground)
   - Or leave it completely white
   - NO detailed environment elements

CRITICAL RULES:
- ABSOLUTELY NO TEXT of any kind (no names, no labels, no boxes)
- Each character = simple stick figure filled with ONE solid bright color
- Each artifact = simple shape filled with ONE solid bright color
- Colors are for identification in later stages
- Maximum simplicity - kindergarten-level stick figures
- Use EXACT colors specified (BRIGHT RED, BRIGHT BLUE, BRIGHT GREEN, etc.)
- Clean white background

Return ONLY the complete sketch prompt text, nothing else."""

        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=prompt
        )
        
        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")
            
        return response.text.strip()
    
    try:
        return _generate()
    except Exception as e:
        st.error(f"Failed to generate sketch prompt: {str(e)}")
        return None


def generate_image(prompt: str, api_token: str) -> Optional[str]:
    """
    Generate an image using Qwen Image API.
    
    Args:
        prompt: Text prompt for image generation
        api_token: Replicate API token
        
    Returns:
        URL of generated image or None if failed
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        
        output = replicate.run(
            QWEN_IMAGE_MODEL,
            input={
                "prompt": prompt
            }
        )
        
        # Handle different output formats
        if isinstance(output, list) and len(output) > 0:
            return str(output[0])
        elif isinstance(output, str):
            return output
        else:
            return str(output)
            
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None


def edit_image(image_url: str, prompt: str, api_token: str, reference_images: Optional[List[str]] = None) -> Optional[str]:
    """
    Edit/refine an image using Qwen Image Edit Plus with optional reference images.
    
    Args:
        image_url: URL or path of input image to edit
        prompt: Text prompt for image editing
        api_token: Replicate API token
        reference_images: Optional list of reference image URLs (for assets)
        
    Returns:
        URL of edited image or None if failed
    """
    try:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        
        # Build image array: main image first, then reference images
        image_array = [image_url]
        if reference_images:
            image_array.extend(reference_images)
        
        output = replicate.run(
            QWEN_IMAGE_EDIT_MODEL,
            input={
                "image": image_array,
                "prompt": prompt
            }
        )
        
        # Handle different output formats
        if isinstance(output, list) and len(output) > 0:
            return str(output[0])
        elif isinstance(output, str):
            return output
        else:
            return str(output)
            
    except Exception as e:
        st.error(f"Image editing failed: {str(e)}")
        return None


def create_pdf(title: str, scenes: List[str], final_images: List[bytes]) -> bytes:
    """
    Create a PDF storybook with alternating image and text pages.
    
    Args:
        title: Story title
        scenes: List of scene descriptions
        final_images: List of image bytes
        
    Returns:
        PDF as bytes
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    
    # Page 1: Title Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 36)
    pdf.set_y(130)
    pdf.multi_cell(0, 20, title, align='C')
    
    # Add scene pages (alternating image and text)
    for idx, (scene_text, image_bytes) in enumerate(zip(scenes, final_images)):
        # Image Page
        pdf.add_page()
        
        # Save image temporarily
        temp_img_path = f"/tmp/scene_{idx}.png"
        try:
            with open(temp_img_path, 'wb') as f:
                f.write(image_bytes)
            
            # Calculate centering
            x = (PDF_WIDTH - IMAGE_WIDTH) / 2
            y = (PDF_HEIGHT - IMAGE_HEIGHT) / 2
            
            pdf.image(temp_img_path, x=x, y=y, w=IMAGE_WIDTH, h=IMAGE_HEIGHT)
            
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
        except Exception as e:
            st.warning(f"Could not add image for scene {idx + 1}: {str(e)}")
        
        # Text Page
        pdf.add_page()
        pdf.set_font("Arial", "", 14)
        pdf.set_y(100)
        pdf.multi_cell(0, 10, f"Scene {idx + 1}\n\n{scene_text}", align='C')
    
    # Return PDF as bytes
    return bytes(pdf.output())


# ============================================================================
# PHASE FUNCTIONS
# ============================================================================

def render_phase1_inputs():
    """Phase 1: Story Inputs - Collect all story parameters."""
    st.header("📝 Phase 1: Story Setup")
    
    # Story Title
    st.subheader("Story Information")
    title = st.text_input(
        "Story Title",
        value=st.session_state.story_title,
        placeholder="Enter your story title..."
    )
    st.session_state.story_title = title
    
    # Scenes
    st.subheader("Story Scenes")
    st.caption("Add descriptions for each scene in your story")
    
    num_scenes = st.number_input(
        "Number of Scenes",
        min_value=1,
        max_value=10,
        value=len(st.session_state.scenes),
        disabled=st.session_state.scenes_analyzed
    )
    
    # Adjust scenes list
    while len(st.session_state.scenes) < num_scenes:
        st.session_state.scenes.append("")
    while len(st.session_state.scenes) > num_scenes:
        st.session_state.scenes.pop()
    
    for i in range(int(num_scenes)):
        st.session_state.scenes[i] = st.text_area(
            f"Scene {i + 1}",
            value=st.session_state.scenes[i],
            placeholder=f"Describe what happens in scene {i + 1}...",
            key=f"scene_{i}",
            disabled=st.session_state.scenes_analyzed
        )
    
    # Analyze Scenes Button
    if not st.session_state.scenes_analyzed:
        st.markdown("---")
        if st.button("🤖 Analyze Scenes with AI", type="primary", use_container_width=True):
            if not st.session_state.gemini_api_key:
                st.error("⚠️ GEMINI_API_KEY not found! Please set it in your .envrc file.")
                return
            
            if not st.session_state.story_title or not st.session_state.story_title.strip():
                st.warning("⚠️ Please enter a story title first.")
                return
            
            if not any(scene.strip() for scene in st.session_state.scenes):
                st.warning("⚠️ Please add at least one scene description.")
                return
            
            with st.spinner("🧠 Analyzing scenes and extracting story elements..."):
                result = analyze_scenes_with_gemini(st.session_state.scenes, st.session_state.gemini_api_key)
                
                if result:
                    # Update characters
                    st.session_state.characters = result.get('characters', [])
                    if not st.session_state.characters:
                        st.session_state.characters = [""]
                    
                    # Update artifacts
                    st.session_state.artifacts = result.get('artifacts', [])
                    if not st.session_state.artifacts:
                        st.session_state.artifacts = [""]
                    
                    # Update environments  
                    st.session_state.environments = result.get('environments', [])
                    if not st.session_state.environments:
                        st.session_state.environments = [""]
                    
                    # Automatically assign unique colors to characters and artifacts
                    st.session_state.character_colors = {}
                    st.session_state.artifact_colors = {}
                    color_idx = 0
                    
                    # Assign colors to characters
                    for i, char in enumerate(st.session_state.characters):
                        if char.strip():
                            st.session_state.character_colors[i] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
                            color_idx += 1
                    
                    # Assign colors to artifacts
                    for i, artifact in enumerate(st.session_state.artifacts):
                        if artifact.strip():
                            st.session_state.artifact_colors[i] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
                            color_idx += 1
                    
                    st.session_state.scenes_analyzed = True
                    st.success("✅ Scenes analyzed! Colors assigned to assets. Review below.")
                    st.rerun()
    
    # Asset Lists - Only show after analysis
    if st.session_state.scenes_analyzed and not st.session_state.assets_confirmed:
        st.markdown("---")
        st.subheader("🎭 Story Assets (AI Generated - Edit as Needed)")
        st.info("Review and edit the automatically extracted story elements. Click 'Confirm & Continue' when ready.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("Characters (with assigned colors)")
            num_chars = st.number_input("# Characters", 1, 10, len(st.session_state.characters), key="num_chars")
            while len(st.session_state.characters) < num_chars:
                st.session_state.characters.append("")
            while len(st.session_state.characters) > num_chars:
                st.session_state.characters.pop()
            
            for i in range(int(num_chars)):
                # Assign color if not yet assigned
                if i not in st.session_state.character_colors and st.session_state.characters[i].strip():
                    color_idx = len(st.session_state.character_colors) + len(st.session_state.artifact_colors)
                    st.session_state.character_colors[i] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
                
                color_badge = f"🎨 {st.session_state.character_colors.get(i, 'N/A')}" if i in st.session_state.character_colors else ""
                st.session_state.characters[i] = st.text_input(
                    f"Character {i + 1} {color_badge}",
                    value=st.session_state.characters[i],
                    placeholder="Character name and description",
                    key=f"char_{i}"
                )
        
        with col2:
            st.caption("Artifacts (with assigned colors)")
            num_artifacts = st.number_input("# Artifacts", 0, 10, len(st.session_state.artifacts), key="num_artifacts")
            while len(st.session_state.artifacts) < num_artifacts:
                st.session_state.artifacts.append("")
            while len(st.session_state.artifacts) > num_artifacts:
                st.session_state.artifacts.pop()
            
            for i in range(int(num_artifacts)):
                # Assign color if not yet assigned
                if i not in st.session_state.artifact_colors and st.session_state.artifacts[i].strip():
                    color_idx = len(st.session_state.character_colors) + len(st.session_state.artifact_colors)
                    st.session_state.artifact_colors[i] = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
                
                color_badge = f"🎨 {st.session_state.artifact_colors.get(i, 'N/A')}" if i in st.session_state.artifact_colors else ""
                st.session_state.artifacts[i] = st.text_input(
                    f"Artifact {i + 1} {color_badge}",
                    value=st.session_state.artifacts[i],
                    placeholder="Artifact description",
                    key=f"artifact_{i}"
                )
        
        with col3:
            st.caption("Environments")
            num_envs = st.number_input("# Environments", 0, 10, len(st.session_state.environments), key="num_envs")
            while len(st.session_state.environments) < num_envs:
                st.session_state.environments.append("")
            while len(st.session_state.environments) > num_envs:
                st.session_state.environments.pop()
            
            for i in range(int(num_envs)):
                st.session_state.environments[i] = st.text_input(
                    f"Environment {i + 1}",
                    value=st.session_state.environments[i],
                    placeholder="Environment description",
                    key=f"env_{i}"
                )
        
        st.markdown("---")
        if st.button("✅ Confirm & Continue to Phase 2", type="primary", use_container_width=True):
            st.session_state.assets_confirmed = True
            st.success("✅ Assets confirmed! You can now proceed to Phase 2: Draft Sketches.")
            st.rerun()
    
    # Show confirmation message if both steps are done
    if st.session_state.scenes_analyzed and st.session_state.assets_confirmed:
        st.markdown("---")
        st.success("✅ Phase 1 Complete! Proceed to Phase 2: Draft Sketches to begin generating images.")
        
        # Option to reset and edit
        if st.button("🔄 Edit Story Setup"):
            st.session_state.scenes_analyzed = False
            st.session_state.assets_confirmed = False
            st.rerun()


def render_phase2_draft_generation():
    """Phase 2: Draft Generation - Create stick figure sketches."""
    st.header("✏️ Phase 2: Draft Sketches")
    
    st.info(
        "🎨 **About Draft Sketches:** These will be SIMPLE COLOR-CODED stick figure sketches. "
        "Each character and artifact is filled with a unique BRIGHT color for identification. "
        "NO text, NO labels, NO name tags - just colored stick figures showing positions and poses. "
        "In Phase 4, the AI uses these colors to match each stick figure/object to the correct detailed asset."
    )
    
    if not st.session_state.api_token:
        st.error("⚠️ REPLICATE_API_TOKEN not found! Please set it in your .envrc file and restart the app.")
        st.info("💡 Run: `source .envrc` or use `./run.sh` to load environment variables.")
        return
    
    if not st.session_state.gemini_api_key:
        st.error("⚠️ GEMINI_API_KEY not found! Please set it in your .envrc file and restart the app.")
        st.info("💡 Run: `source .envrc` or use `./run.sh` to load environment variables.")
        return
    
    if not st.session_state.story_title or not any(st.session_state.scenes):
        st.warning("⚠️ Please complete Phase 1 first.")
        return
    
    if not st.session_state.assets_confirmed:
        st.warning("⚠️ Please analyze your scenes with AI and confirm assets in Phase 1 before proceeding.")
        return
    
    if st.button("🎨 Generate Draft Sketches", type="primary"):
        st.session_state.draft_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, scene in enumerate(st.session_state.scenes):
            if not scene.strip():
                continue
            
            status_text.text(f"Analyzing scene {idx + 1}/{len(st.session_state.scenes)}...")
            
            # Use Gemini to generate precise sketch prompt based on scene and characters
            with st.spinner(f"Analyzing scene {idx + 1} with AI..."):
                draft_prompt = generate_sketch_prompt_with_gemini(
                    scene, 
                    st.session_state.characters,
                    st.session_state.artifacts,
                    st.session_state.character_colors,
                    st.session_state.artifact_colors,
                    st.session_state.gemini_api_key
                )
                
                if not draft_prompt:
                    st.warning(f"⚠️ Could not generate prompt for scene {idx + 1}. Skipping...")
                    continue
            
            status_text.text(f"Generating sketch for scene {idx + 1}/{len(st.session_state.scenes)}...")
            
            with st.spinner(f"Creating sketch for scene {idx + 1}..."):
                image_url = generate_image(draft_prompt, st.session_state.api_token)
                
                if image_url:
                    # Download and store image
                    image_bytes = download_image(image_url)
                    if image_bytes:
                        st.session_state.draft_images.append({
                            'url': image_url,
                            'bytes': image_bytes,
                            'scene_idx': idx,
                            'prompt': draft_prompt  # Store the prompt for reference
                        })
                    time.sleep(5)  # Rate limiting - increased to avoid quota exhaustion
            
            progress_bar.progress((idx + 1) / len(st.session_state.scenes))
        
        status_text.text("✅ Draft sketches complete!")
        st.success(f"Generated {len(st.session_state.draft_images)} draft sketches!")
    
    # Display draft images
    if st.session_state.draft_images:
        st.subheader("Draft Sketches Preview")
        st.info("💡 You can edit the AI-generated prompts and regenerate individual scenes if needed.")
        
        for idx, draft in enumerate(st.session_state.draft_images):
            scene_idx = draft['scene_idx']
            
            st.markdown(f"### Scene {scene_idx + 1}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(draft['bytes'], use_column_width=True)
            
            with col2:
                st.caption(f"**Scene Description:**")
                scene_text = st.session_state.scenes[scene_idx]
                if len(scene_text) > 200:
                    st.text(scene_text[:200] + "...")
                else:
                    st.text(scene_text)
                
                # Editable prompt
                if 'prompt' in draft:
                    st.caption("**AI-Generated Sketch Prompt (Editable):**")
                    edited_prompt = st.text_area(
                        "Edit prompt if needed",
                        value=draft['prompt'],
                        height=150,
                        key=f"prompt_edit_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    # Regenerate button
                    if st.button(f"🔄 Regenerate Scene {scene_idx + 1}", key=f"regen_{idx}", type="secondary"):
                        api_token = st.session_state.api_token
                        if not api_token:
                            st.error("API token not found!")
                            return
                        
                        if not edited_prompt or not edited_prompt.strip():
                            st.error("Prompt cannot be empty!")
                            return
                        
                        with st.spinner(f"Regenerating scene {scene_idx + 1}..."):
                            # Use the edited prompt
                            new_image_url = generate_image(edited_prompt, api_token)
                            
                            if new_image_url:
                                new_image_bytes = download_image(new_image_url)
                                if new_image_bytes:
                                    # Update the draft image
                                    st.session_state.draft_images[idx]['url'] = new_image_url
                                    st.session_state.draft_images[idx]['bytes'] = new_image_bytes
                                    st.session_state.draft_images[idx]['prompt'] = edited_prompt
                                    st.success(f"✅ Scene {scene_idx + 1} regenerated!")
                                    st.rerun()
                            else:
                                st.error(f"Failed to regenerate scene {scene_idx + 1}")
            
            if idx < len(st.session_state.draft_images) - 1:
                st.markdown("---")


def render_phase3_asset_design():
    """Phase 3: Asset Design - Generate high-quality asset references."""
    st.header("🎨 Phase 3: Asset Design")
    
    st.info(
        "🎭 **About Asset Design:** Generate detailed reference sheets for characters and artifacts. "
        "Characters and artifacts will be shown as 3-way turnaround views (front, side, back) "
        "for consistent reference across your storybook scenes."
    )
    
    if not st.session_state.api_token:
        st.error("⚠️ REPLICATE_API_TOKEN not found! Please set it in your .envrc file and restart the app.")
        st.info("💡 Run: `source .envrc` or use `./run.sh` to load environment variables.")
        return
    
    # Collect all assets
    all_assets = []
    for char in st.session_state.characters:
        if char.strip():
            all_assets.append(('character', char))
    for artifact in st.session_state.artifacts:
        if artifact.strip():
            all_assets.append(('artifact', artifact))
    for env in st.session_state.environments:
        if env.strip():
            all_assets.append(('environment', env))
    
    if not all_assets:
        st.info("No assets defined. Add characters, artifacts, or environments in Phase 1.")
        return
    
    if st.button("🖼️ Design Assets", type="primary"):
        st.session_state.asset_designs = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (asset_type, asset_desc) in enumerate(all_assets):
            status_text.text(f"Designing {asset_type}: {asset_desc[:30]}...")
            
            # Create detailed prompt based on asset type
            if asset_type == 'character':
                prompt = (
                    f"Professional character turnaround sheet showing THREE views of the same character: "
                    f"front view, side view, and back view. High quality character design, detailed concept art style. "
                    f"Character description: {asset_desc}. "
                    f"Show the character standing in neutral pose from all three angles side by side. "
                    f"Clean neutral background (white or light gray), no environment elements. "
                    f"Consistent design across all three views. Professional animation reference style."
                )
            elif asset_type == 'artifact':
                prompt = (
                    f"Professional object turnaround sheet showing THREE views of the same item: "
                    f"front view, side view, and back/rear view. High quality product design, detailed technical illustration. "
                    f"Object description: {asset_desc}. "
                    f"Show the object clearly from all three angles side by side. "
                    f"Clean neutral background (white or light gray), no other elements. "
                    f"Consistent design across all three views. Professional technical illustration / product design style."
                )
            else:  # environment
                prompt = f"Detailed environment design, high quality realistic setting, professional looks. Environment: {asset_desc}"
            
            with st.spinner(f"Creating {asset_type} design..."):
                image_url = generate_image(prompt, st.session_state.api_token)
                
                if image_url:
                    image_bytes = download_image(image_url)
                    if image_bytes:
                        key = f"{asset_type}_{idx}"
                        st.session_state.asset_designs[key] = {
                            'type': asset_type,
                            'description': asset_desc,
                            'url': image_url,
                            'bytes': image_bytes
                        }
                    time.sleep(1)  # Rate limiting
            
            progress_bar.progress((idx + 1) / len(all_assets))
        
        status_text.text("✅ Asset designs complete!")
        st.success(f"Created {len(st.session_state.asset_designs)} asset designs!")
    
    # Display asset designs
    if st.session_state.asset_designs:
        st.subheader("Asset Design Gallery")
        st.caption("Turnaround sheets showing multiple views for consistent reference")
        
        # Display in 2 columns since turnaround sheets are wider
        cols = st.columns(2)
        for idx, (key, asset) in enumerate(st.session_state.asset_designs.items()):
            with cols[idx % 2]:
                asset_type_display = asset['type'].title()
                if asset['type'] in ['character', 'artifact']:
                    caption = f"{asset_type_display} Turnaround: {asset['description'][:40]}..."
                else:
                    caption = f"{asset_type_display}: {asset['description'][:40]}..."
                st.image(asset['bytes'], caption=caption, use_column_width=True)


def render_phase4_final_composition():
    """Phase 4: Final Composition - Refine drafts with asset details."""
    st.header("✨ Phase 4: Final Scene Rendering")
    
    st.info(
        "✨ **About Final Rendering:** This phase uses COLOR IDENTIFICATION to transform your sketches. "
        "The AI identifies each colored stick figure/object in the draft sketch and replaces it with "
        "the matching detailed asset from Phase 3. For example: a BRIGHT RED stick figure becomes the "
        "full character design for that character, a BRIGHT BLUE object becomes the detailed artifact. "
        "The composition and poses are preserved, but stick figures become beautiful illustrations."
    )
    
    if not st.session_state.api_token:
        st.error("⚠️ REPLICATE_API_TOKEN not found! Please set it in your .envrc file and restart the app.")
        st.info("💡 Run: `source .envrc` or use `./run.sh` to load environment variables.")
        return
    
    if not st.session_state.draft_images:
        st.warning("⚠️ Please generate draft sketches first (Phase 2).")
        return
    
    if not st.session_state.asset_designs:
        st.warning("⚠️ Please design assets first (Phase 3) to use as references.")
        return
    
    if st.button("🎬 Render Final Scenes", type="primary"):
        st.session_state.final_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, draft in enumerate(st.session_state.draft_images):
            scene_idx = draft['scene_idx']
            scene_text = st.session_state.scenes[scene_idx]
            draft_sketch_url = draft['url']
            
            status_text.text(f"Collecting assets for scene {idx + 1}...")
            
            # Collect all asset images with their identifying labels AND colors
            # Assets are stored with sequential indices across all types
            asset_references = []
            asset_labels = []
            color_mappings = []  # Maps color to asset name
            asset_idx = 0  # Global index counter matching Phase 3 storage
            
            # Add character assets with color mappings
            for char_idx, char_desc in enumerate(st.session_state.characters):
                if char_desc.strip():
                    key = f"character_{asset_idx}"
                    if key in st.session_state.asset_designs:
                        asset_url = st.session_state.asset_designs[key]['url']
                        asset_references.append(asset_url)
                        # Extract character name (before colon)
                        char_name = char_desc.split(':')[0].strip()
                        asset_labels.append(f"CHARACTER: {char_name}")
                        
                        # Add color mapping
                        if char_idx in st.session_state.character_colors:
                            color = st.session_state.character_colors[char_idx]
                            color_mappings.append(f"{color} figure → {char_name} CHARACTER asset")
                    asset_idx += 1
            
            # Add artifact assets with color mappings
            for art_idx, art_desc in enumerate(st.session_state.artifacts):
                if art_desc.strip():
                    key = f"artifact_{asset_idx}"
                    if key in st.session_state.asset_designs:
                        asset_url = st.session_state.asset_designs[key]['url']
                        asset_references.append(asset_url)
                        # Extract artifact name (before colon)
                        art_name = art_desc.split(':')[0].strip()
                        asset_labels.append(f"ARTIFACT: {art_name}")
                        
                        # Add color mapping
                        if art_idx in st.session_state.artifact_colors:
                            color = st.session_state.artifact_colors[art_idx]
                            color_mappings.append(f"{color} object → {art_name} ARTIFACT asset")
                    asset_idx += 1
            
            # Add environment asset
            for env_desc in st.session_state.environments:
                if env_desc.strip():
                    key = f"environment_{asset_idx}"
                    if key in st.session_state.asset_designs:
                        asset_url = st.session_state.asset_designs[key]['url']
                        asset_references.append(asset_url)
                        # Extract environment name
                        env_name = env_desc.split(':')[0].strip() if ':' in env_desc else env_desc[:30]
                        asset_labels.append(f"ENVIRONMENT: {env_name}")
                    asset_idx += 1
            
            status_text.text(f"Rendering final scene {idx + 1}/{len(st.session_state.draft_images)}...")
            
            # Build refinement prompt that uses the assets with color mappings
            asset_list = "\\n".join(asset_labels) if asset_labels else "No specific assets"
            color_map_text = "\\n".join(color_mappings) if color_mappings else "No color mappings"
            
            refinement_prompt = (
                f"Transform this COLOR-CODED stick figure sketch into a beautiful, detailed, full-color children's storybook illustration.\\n\\n"
                f"SCENE: {scene_text}\\n\\n"
                f"COLOR-TO-ASSET MAPPING (use this to identify which sketch element matches which asset):\\n{color_map_text}\\n\\n"
                f"AVAILABLE ASSETS TO USE:\\n{asset_list}\\n\\n"
                f"INSTRUCTIONS:\\n"
                f"1. The sketch uses SOLID COLORS to identify different characters and objects\\n"
                f"2. Use the color mapping above to match each colored element to its corresponding asset\\n"
                f"3. Replace each colored stick figure/object with the detailed asset that matches its color\\n"
                f"4. Keep the exact same composition, layout, and positions as the sketch\\n"
                f"5. Maintain character poses and gestures from the sketch\\n"
                f"6. The colored sketch shows WHAT goes WHERE - the assets show HOW they should look\\n"
                f"7. Create a fully rendered, professional children's book illustration\\n"
                f"8. Use warm colors, soft lighting, clean style suitable for ages 5-10\\n"
                f"9. Add an appropriate background matching the environment asset provided\\n\\n"
                f"CRITICAL: Match colors EXACTLY - BRIGHT RED figure → use the asset labeled as BRIGHT RED!"
            )
            
            with st.spinner(f"Creating scene {idx + 1}..."):
                # Use edit_image with draft sketch as base and all assets as references
                image_url = edit_image(
                    draft_sketch_url, 
                    refinement_prompt, 
                    st.session_state.api_token,
                    reference_images=asset_references if asset_references else None
                )
                
                if image_url:
                    image_bytes = download_image(image_url)
                    if image_bytes:
                        st.session_state.final_images.append({
                            'url': image_url,
                            'bytes': image_bytes,
                            'scene_idx': scene_idx
                        })
                    time.sleep(1)  # Rate limiting
            
            progress_bar.progress((idx + 1) / len(st.session_state.draft_images))
        
        status_text.text("✅ Final rendering complete!")
        st.success(f"Rendered {len(st.session_state.final_images)} final scenes!")
    
    # Display final images
    if st.session_state.final_images:
        st.subheader("Final Scenes Preview")
        
        cols = st.columns(2)
        for idx, final in enumerate(st.session_state.final_images):
            with cols[idx % 2]:
                st.image(final['bytes'], caption=f"Final Scene {final['scene_idx'] + 1}", 
                        use_column_width=True)


def render_phase5_pdf_compilation():
    """Phase 5: PDF Compilation - Create downloadable storybook."""
    st.header("📚 Phase 5: PDF Storybook")
    
    if not st.session_state.final_images:
        st.warning("⚠️ Please render final scenes first (Phase 4).")
        return
    
    if not st.session_state.story_title:
        st.warning("⚠️ Please enter a story title in Phase 1.")
        return
    
    st.info(f"Ready to compile {len(st.session_state.final_images)} scenes into a PDF storybook.")
    
    if st.button("📖 Generate PDF Storybook", type="primary"):
        with st.spinner("Compiling PDF... This may take a moment..."):
            try:
                # Extract scene texts and image bytes in order
                final_scenes = []
                final_image_bytes = []
                
                for final_img in st.session_state.final_images:
                    scene_idx = final_img['scene_idx']
                    final_scenes.append(st.session_state.scenes[scene_idx])
                    final_image_bytes.append(final_img['bytes'])
                
                # Create PDF
                pdf_bytes = create_pdf(
                    st.session_state.story_title,
                    final_scenes,
                    final_image_bytes
                )
                
                # Encode for download
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                
                # Success message
                st.success("✅ PDF generated successfully!")
                
                # Download button
                st.download_button(
                    label="⬇️ Download Storybook PDF",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.story_title.replace(' ', '_')}_storybook.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Failed to generate PDF: {str(e)}")
                st.exception(e)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Storybook Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar - API Status (without showing the actual token)
    with st.sidebar:
        st.subheader("🔧 Configuration Status")
        if st.session_state.api_token:
            st.success("✅ API Token Loaded")
            st.caption("Token loaded from environment")
        else:
            st.error("❌ API Token Missing")
            st.caption("Set REPLICATE_API_TOKEN in .envrc")
            st.code("source .envrc", language="bash")
    
    # Title
    st.title("📚 AI-Powered Storybook Generator")
    st.markdown("*Create beautiful illustrated storybooks using Qwen AI Models*")
    st.markdown("---")
    
    # Phase selection tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Story Setup",
        "✏️ Draft Sketches",
        "🎨 Asset Design",
        "✨ Final Rendering",
        "📚 PDF Export"
    ])
    
    with tab1:
        render_phase1_inputs()
    
    with tab2:
        render_phase2_draft_generation()
    
    with tab3:
        render_phase3_asset_design()
    
    with tab4:
        render_phase4_final_composition()
    
    with tab5:
        render_phase5_pdf_compilation()
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Qwen Image & Qwen Image Edit Plus AI Models via Replicate")


if __name__ == "__main__":
    main()
