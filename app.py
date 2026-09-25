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

# ── Set up page configuration ──
st.set_page_config(
    page_title="Magic Picture Storybook",
    page_icon="🎈",
    layout="wide"
)


# ==============================================================================
# Function part: Core pipeline modules (No caching, purely direct calls)
# ==============================================================================

def img2text(image: Image.Image, model_name: str = "Salesforce/blip-image-captioning-base") -> str:
    """Stage 1: Generate a concise descriptive caption from an input image."""
    # 直接在函式內部建立流水線
    captioner = pipeline("image-to-text", model=model_name)
    results = captioner(image)
    return results[0]["generated_text"]


def text2story(scenario: str, model_name: str = "roneneldan/TinyStories-33M", min_words: int = 50, max_words: int = 80) -> str:
    """Stage 2: Expand the image scenario into a 50-100 word child-friendly story."""
    # 直接在函式內部建立故事生成流水線
    story_generator = pipeline("text-generation", model=model_name)
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
    Returns a BytesIO stream ready for direct web playback.
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
    # 1. Centered Header
    st.markdown("<h1 style='text-align: center;'>🎈 Magic Picture Storybook</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Drop a picture below and watch the bedtime story unfold automatically!</p>", unsafe_allow_html=True)
    st.divider()

    # 2. Pipeline Configuration Variables
    caption_model = "Salesforce/blip-image-captioning-base"
    story_model = "roneneldan/TinyStories-33M"
    min_story_words = 50
    max_story_words = 80
    audio_accent = "co.uk"

    # 3. Two-Column Dashboard Layout (Zero-scroll design)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📸 Choose a Picture")
        uploaded_image_file = st.file_uploader(
            "Drop your photo here:",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_image_file is not None:
            image = Image.open(uploaded_image_file)
            st.image(image, caption="🌟 Uploaded Picture", width=380)

    with col_right:
        if uploaded_image_file is not None:
            with st.spinner("🧙‍♂️ Magic is happening... crafting your story & audio..."):
                # Stage 1: Captioning
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

            # Word count validation
            word_count = len(story_text.split())
            st.caption(f"📏 Story word count: approximately {word_count} words.")

            # Audio Deliverable
            st.subheader("🔊 Listen Along:")
            st.audio(audio_data, format="audio/mp3", autoplay=True)
        else:
            st.info("👈 Drop an image on the left, and your story will appear here instantly!")


# Execution entry point
if __name__ == "__main__":
    main()
