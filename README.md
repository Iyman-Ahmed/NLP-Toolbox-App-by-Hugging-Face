# NLP ToolBox App (Hugging Face + Streamlit)

An interactive NLP playground built with Hugging Face Transformers, Streamlit, and Pyngrok that lets you experiment with:

- Text Summarization  
- Sentiment Analysis  
- Language Translation  

This project is ideal for learning and experimenting with Hugging Face pipelines and deploying them via a simple web interface.

---

## Features

- **Summarization**  
  Choose from state-of-the-art summarization models (`facebook/bart-large-cnn`, `t5-small`, `google/pegasus-xsum`) and tune hyperparameters such as:
  - `max_length`
  - `min_length`
  - `temperature` (creativity in output)

- **Sentiment Analysis**  
  Uses `distilbert-base-uncased-finetuned-sst-2-english` to classify text as Positive or Negative with confidence scores.

- **Translation**  
  Supports translating English text to:
  - German (`Helsinki-NLP/opus-mt-en-de`)
  - French (`Helsinki-NLP/opus-mt-en-fr`)
  - Spanish (`Helsinki-NLP/opus-mt-en-es`)

- **User-Friendly UI**  
  - Sidebar for model selection & hyperparameter tuning  
  - Expanders for extra information (model name, confidence score)  
  - Progress bar for sentiment confidence  
  - Loading spinner while models process your input  

---

## Tech Stack

- [Streamlit](https://streamlit.io/): For building the interactive web UI  
- [Hugging Face Transformers](https://huggingface.co/transformers): For NLP pipelines  
- [Pyngrok](https://pyngrok.readthedocs.io/): For creating public URLs in cloud environments  
- Python 3.10+

---

## Installation & Setup

Follow these steps to run the project locally or in GitHub Codespaces:

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NLP-Toolbox-App-by-Hugging-Face.git
cd NLP-Toolbox-App-by-Hugging-Face

2. Create and Activate Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. (Optional) Configure Ngrok

Get your Ngrok token from ngrok dashboard
 and set it inside app.py:

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
