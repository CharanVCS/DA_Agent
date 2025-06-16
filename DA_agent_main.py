import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from docx import Document
import PyPDF2
import together
import tempfile
import json
import os

# Optional: Use dotenv for security
# from dotenv import load_dotenv
# load_dotenv()

# Secure API Key setup
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY') or '4d808f2b2135fe4c2c3a42ef1c9e47d1bc48921f1a7eb11cef1e4d40c1960a0b'
together.api_key = TOGETHER_API_KEY

st.title('Data Analyst Agent')
st.write('Upload a document (.docx, .txt, .xlsx, .csv, .pdf) and ask questions about your data. Powered by Together.ai.')

SUPPORTED_TYPES = ['csv', 'xlsx', 'xls', 'txt', 'doc', 'docx', 'pdf', 'json']

def load_file(file):
    ext = file.name.lower().split('.')[-1]
    try:
        if ext == 'csv':
            return pd.read_csv(file)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(file)
        elif ext == 'txt':
            return file.read().decode('utf-8')
        elif ext in ['doc', 'docx']:
            return read_docx(file)
        elif ext == 'pdf':
            return read_pdf(file)
        elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif']:
            return Image.open(file)
        elif ext == 'json':
            return json.load(file)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

def read_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(file.read())
        tmp.flush()
        doc = Document(tmp.name)
        os.unlink(tmp.name)
    return '\n'.join([p.text for p in doc.paragraphs])

def read_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file.read())
        tmp.flush()
        reader = PyPDF2.PdfReader(tmp.name)
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        os.unlink(tmp.name)
    return text

def ask_llm(prompt, system_prompt=None):
    try:
        response = together.Complete.create(
            model=selected_model,
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
            stop=None,
            system=system_prompt
        )
        return response['choices'][0]['text'].strip() if 'choices' in response else "Invalid API response."
    except Exception as e:
        return f"API Error: {e}"

def generate_summary(file, content):
    ext = file.name.split('.')[-1].lower()
    if isinstance(content, pd.DataFrame):
        preview = content.head(50).to_csv(index=False)
        return ask_llm(f"Summarize this dataset:\n{preview}\nin a single paragraph")
    elif isinstance(content, dict):
        sample = json.dumps(content, indent=2)[:500]
        return ask_llm(f"Summarize this JSON data:\n{sample}")
    elif isinstance(content, Image.Image):
        return ask_llm("Analyze this image and summarize its contents.")
    elif isinstance(content, str):
        return ask_llm(f"Summarize this text:\n{content[:1000]}")
    return "Unsupported content type."

def display_visualization(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for visualization.")
        return
    selected_col = st.selectbox("Select numeric column for histogram", numeric_cols)
    if selected_col:
        fig, ax = plt.subplots()
        sns.histplot(data[selected_col], ax=ax, kde=True)
        ax.set_title(f"Histogram of {selected_col}")
        st.pyplot(fig)

# --- App UI ---
uploaded_file = st.file_uploader("Upload your file", type=SUPPORTED_TYPES)
model_options = [
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1',
    'Qwen/Qwen3-235B-A22B-fp8-tput',
    'meta-llama/Llama-Vision-Free',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'
]
selected_model = st.selectbox("Choose LLM model", model_options, index=1)

if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'file_data' not in st.session_state:
    st.session_state.file_data = None

if uploaded_file:
    try:
        file_data = load_file(uploaded_file)
        st.session_state.file_data = file_data
        st.success("File loaded successfully!")

        if isinstance(file_data, pd.DataFrame):
            st.dataframe(file_data.head())
        elif isinstance(file_data, Image.Image):
            st.image(file_data, caption="Uploaded Image", use_column_width=True)
        else:
            st.text_area("Text Preview", str(file_data)[:1000], height=200)

    except Exception as e:
        st.error(str(e))

if st.button("Analyze Data"):
    if uploaded_file and not st.session_state.file_data.empty:
        with st.spinner("Generating summary..."):
            st.session_state.summary = generate_summary(uploaded_file, st.session_state.file_data)
    else:
        st.warning("Please upload a file first.")

if st.session_state.summary:
    st.markdown(f"### 🔍 Summary:\n{st.session_state.summary}")
    if isinstance(st.session_state.file_data, pd.DataFrame):
        st.subheader("📊 Data Visualization")
        display_visualization(st.session_state.file_data)

# --- Ask Questions ---
user_question = st.text_input("Ask a question about your data:")
if user_question:
    content = st.session_state.file_data
    if not content:
        st.warning("Please upload and analyze a file first.")
    else:
        context = ''
        if isinstance(content, pd.DataFrame):
            context = content.head(20).to_csv(index=False)
        elif isinstance(content, str):
            context = content[:2000]
        else:
            context = 'Image data loaded.'
        prompt = f"Given this data:\n{context}\nAnswer the following:\n{user_question}"
        with st.spinner("Thinking..."):
            answer = ask_llm(prompt)
        st.markdown(f"**Agent:** {answer}")
