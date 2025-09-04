from transformers import pipeline
import streamlit as st
from pyngrok import ngrok
import subprocess

# --- Streamlit Page Config ---
st.set_page_config(page_title="NLP ToolBox", layout="centered")
st.title("🧠 NLP ToolBox App by Hugging Face")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
selected_feature = st.sidebar.radio("Choose a Feature:", [
    'Sentiment Analysis',
    'Summary',
    'Translation',
])

# --- Model Selection and Hyperparameters ---
selected_model = ""
max_length = 130
min_length = 30
temperature = 1.0

if selected_feature == 'Summary':
    st.sidebar.info("ℹ️ Summarization models may take time to load.")
    selected_model = st.sidebar.selectbox("Choose a Summarization Model", [
        'facebook/bart-large-cnn',
        't5-small',
        'google/pegasus-xsum'
    ])
    max_length = st.sidebar.slider("Max Summary Length", 30, 300, 130)
    min_length = st.sidebar.slider("Min Summary Length", 10, 100, 30)
    temperature = st.sidebar.slider("Temperature (creativity)", 0.0, 2.0, 1.0, 0.1)

elif selected_feature == 'Translation':
    st.sidebar.info("🌍 Translation models may take time to load.")
    selected_model = st.sidebar.selectbox("Choose a Translation Model", [
        'Helsinki-NLP/opus-mt-en-de',
        'Helsinki-NLP/opus-mt-en-fr',
        'Helsinki-NLP/opus-mt-en-es'
    ])

# --- Caching Functions ---
@st.cache_resource
def summariser(model):
    return pipeline('summarization', model=model)

@st.cache_resource
def analyser():
    return pipeline('sentiment-analysis')

@st.cache_resource
def translator(model):
    return pipeline('translation', model=model)

# --- User Input ---
input_text = st.text_area("✍️ Enter Text Here")

if st.button("🚀 Process"):
    if len(input_text.strip()) == 0:
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("⏳ Processing..."):
            if selected_feature == 'Summary':
                processor = summariser(selected_model)
                summary = processor(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=temperature
                )
                st.subheader("📄 Summary")
                st.success(summary[0]['summary_text'])
                with st.expander("🔍 Details"):
                    st.write(f"**Model Used:** {selected_model}")
                    st.write(f"**Max Length:** {max_length}, **Min Length:** {min_length}, **Temperature:** {temperature}")

            elif selected_feature == 'Sentiment Analysis':
                processor = analyser()
                analysis = processor(input_text)
                label, score = analysis[0]['label'], analysis[0]['score']
                st.subheader("📊 Sentiment Analysis")
                st.success(f"**Sentiment:** {label}")
                st.progress(score)
                with st.expander("🔍 Details"):
                    st.write(f"Confidence: {score:.4f}")
                    st.write("Model: `distilbert-base-uncased-finetuned-sst-2-english`")

            elif selected_feature == 'Translation':
                processor = translator(selected_model)
                translation = processor(input_text)
                st.subheader("🌎 Translation")
                st.success(translation[0]['translation_text'])
                with st.expander("🔍 Details"):
                    st.write(f"Model Used: {selected_model}")

# --- Ngrok Tunnel Setup ---
ngrok.kill()
ngrok.set_auth_token("Enter_Your_Token_Here")  # Replace with your ngrok auth token
# public_url = ngrok.connect(8501)
# print(f"Public URL: {public_url}")

# Run Streamlit app (only needed in Codespaces)
subprocess.Popen(["streamlit", "run", "app.py"])

