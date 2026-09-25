"""
ISOM5240 Individual Assignment: Storytelling Application for Children (Aged 3-10)
Description: A child-friendly, zero-scroll, automated storytelling web app with instant narration.
"""

# ==============================================================================
# Import part
# ==============================================================================
import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ── Set up page configuration (Wide layout for a clean, zero-scroll interface) ──
st.set_page_config(
    page_title="Magic Picture Storybook",
    page_icon="🎈",
    layout="wide"
)


# ==============================================================================
# Function part: Core pipeline modules
# ==============================================================================

@st.cache_resource
def load_caption_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Load and cache the image captioning model to optimize runtime efficiency."""
    return pipeline("image-to-text", model=model_name)


@st.cache_resource
def load_story_model(model_name: str = "roneneldan/TinyStories-33M"):
    """Load and cache the lightweight kid-friendly story generation model."""
    return pipeline("text-generation", model=model_name)


def img2text(image: Image.Image, model_name: str = "Salesforce/blip-image-captioning-base") -> str:
    """Stage 1: Generate a concise descriptive caption from an input image."""
    captioner = load_caption_model(model_name=model_name)
    results = captioner(image)
    return results[0]["generated_text"]


def text2story(scenario: str, model_name: str = "roneneldan/TinyStories-33M", min_words: int = 50, max_words: int = 80) -> str:
    """Stage 2: Expand the image scenario into a 50-100 word child-friendly story."""
    story_generator = load_story_model(model_name=model_name)
    kid_prompt = f"Once upon a time, there was {scenario}. "
    
    min_new = int(min_words * 1.1)
    max_new = int(max_words * 1.2)
    
    results = story_generator(
        kid_prompt,
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        do_sample=True,
        temperature=0.7
    )
    raw_story = results[0]["generated_text"]
    return " ".join(raw_story.split())


def text2audio(story_text: str, accent: str = "co.uk") -> io.BytesIO:
    """
    Stage 3: Synthesize speech from story text into an in-memory audio buffer using gTTS.
    Returns a BytesIO stream ready for direct web playback without temporary file pollution.
    """
    tts = gTTS(text=story_text, lang='en', tld=accent, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# ==============================================================================
# Main part: Web application UI layout and automated workflow
# ==============================================================================

def main():
    # 1. Compact Header
    st.title("🎈 Magic Picture Storybook")
    st.caption("Drop a picture below and watch the bedtime story unfold automatically!")

    # 2. Pipeline Configuration Variables
    caption_model = "Salesforce/blip-image-captioning-base"
    story_model = "roneneldan/TinyStories-33M"
    min_story_words = 50
    max_story_words = 80
    audio_accent = "co.uk"  # British accent suitable for HK school curricula

    # 3. Two-Column Dashboard Layout (Zero-scroll design)
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        # Drag-and-Drop Image Uploader
        uploaded_image_file = st.file_uploader(
            "📸 Drop a picture here:",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a picture to instantly trigger the story!"
        )

        if uploaded_image_file is not None:
            image = Image.open(uploaded_image_file)
            st.image(image, caption="🌟 Uploaded Picture", use_container_width=True)

    with col_right:
        if uploaded_image_file is not None:
            # Automated Processing with Compact Spinner
            with st.spinner("🧙‍♂️ Magic is happening... crafting your story & audio..."):
                # Stage 1: Image Captioning
                caption = img2text(image=image, model_name=caption_model)

                # Stage 2: Story Generation
                story_text = text2story(
                    scenario=caption,
                    model_name=story_model,
                    min_words=min_story_words,
                    max_words=max_story_words
                )

                # Stage 3: Audio Synthesis
                audio_data = text2audio(story_text=story_text, accent=audio_accent)

            # Story Deliverable
            st.subheader("📚 Here is Your Story:")
            st.success(story_text)

            # Word count validation for rubric adherence (50-100 words)
            word_count = len(story_text.split())
            st.caption(f"📏 Story word count: approximately {word_count} words.")

            # Audio Deliverable: Automatic playback on completion
            st.subheader("🔊 Listen Along:")
            st.audio(audio_data, format="audio/mp3", autoplay=True)
        else:
            # Prompt guide when empty
            st.info("👈 Drop an image on the left, and your story will appear here instantly!")


# Execution entry point
if __name__ == "__main__":
    main()
