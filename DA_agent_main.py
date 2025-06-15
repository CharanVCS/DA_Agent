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

# Set your Together.ai API key
TOGETHER_API_KEY = '4d808f2b2135fe4c2c3a42ef1c9e47d1bc48921f1a7eb11cef1e4d40c1960a0b'  # Replace with your key
together.api_key = TOGETHER_API_KEY

st.title('Data Analyst Agent')

st.write('Upload a document (.doc, .txt, .xlsx, .csv, .pdf, image) and ask questions about your data. Powered by Together.ai.')

def read_file(file):
    name = file.name.lower()
    ext = os.path.splitext(name)[1]
    if ext in ['.csv', '.xlsx', '.xls']:
        if ext == '.csv':
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    elif ext == '.txt':
        return file.read().decode('utf-8')
    elif ext in ['.doc', '.docx']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp.flush()
            doc = Document(tmp.name)
            os.unlink(tmp.name)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp.flush()
            reader = PyPDF2.PdfReader(tmp.name)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            os.unlink(tmp.name)
        return text
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        return Image.open(file)
    else:
        raise ValueError('Unsupported file type: ' + ext)

def ask_llama(prompt, system_prompt=None):
    try:
        response = together.Complete.create(
            model=selected_model,
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
            stop=None,
            system=system_prompt
        )
        # Defensive: check response structure
        if (
            isinstance(response, dict)
            and 'choices' in response
            and len(response['choices']) > 0
            and 'text' in response['choices'][0]
        ):
            return response['choices'][0]['text'].strip()
        else:
            return f"Unexpected API response: {response}"
    except Exception as e:
        return f"Error from Together API: {e}"

uploaded_file = st.file_uploader('Upload your file', type=['csv', 'xlsx', 'xls', 'txt', 'doc', 'docx', 'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'gif'])


data = None
if uploaded_file:
    try:
        data = read_file(uploaded_file)
        st.success('File loaded successfully!')
        
        if isinstance(data, pd.DataFrame):
            st.write('Preview:')
            st.dataframe(data.head())
        elif isinstance(data, Image.Image):
            st.image(data, caption='Uploaded Image', use_column_width=True)
        else:
            st.text_area('File Content', data[:1000] if isinstance(data, str) else str(data)[:1000], height=200)
    except Exception as e:
        st.error(f'Error loading file: {e}')

model_options = [
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1',
    'Qwen/Qwen3-235B-A22B-fp8-tput',
    'meta-llama/Llama-Vision-Free',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
]
selected_model = st.selectbox('Select Model', model_options, index=1)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'answer' not in st.session_state:
    st.session_state.answer = None


if st.button('Analyze Data'):
    st.session_state.analyzed = True
    def analyze_data(file):
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
            prompt = f"Summarize this dataset:\n{data.head(50).to_csv(index=False)} in a single paragraph"
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
            prompt = f"Summarize this dataset:\n{data.head(50).to_csv(index=False)} in a single paragraph"
        elif file.name.endswith('.json'):
            data = json.load(file)
            prompt = f"Summarize this JSON data:\n{json.dumps(data, indent=2)[:500]} in a single paragraph"
        elif file.name.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file)
            prompt = "Analyze this image and summarize its contents in a single paragraph"
        else:
            prompt = "Unsupported file format"
        
        return ask_llama(prompt)

    with st.spinner("Generating summary..."):
        st.session_state.answer = analyze_data(uploaded_file)
else:
     st.write('Please upload a file and start Analyze data.')

if st.session_state.analyzed:
    st.markdown(f'**Summary:** {st.session_state.answer}')
    if isinstance(data, pd.DataFrame):
        st.subheader('Visualize Data')
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if columns:
            col = st.selectbox('Select column for histogram', columns)
            if col:
                fig, ax = plt.subplots()
                sns.histplot(data=data, x=col, ax=ax)
                ax.set_title(f'Histogram of {col}')
                st.pyplot(fig)
        else:
            st.info('No numeric columns available for visualization.')

user_question = st.text_input('Ask a question about your data:')
if user_question:
    if data is None:
        st.write('Please upload a file and analyze data first.')
    elif st.session_state.analyzed is None:
        st.write('Please analyze data first.')
    else :
        if isinstance(data, pd.DataFrame):
            context = data.head(20).to_csv(index=False)
        elif isinstance(data, str):
            context = data[:2000]
        else:
            context = 'Image file loaded.'
        prompt = f'Given the following data/context:\n{context}\nAnswer the following question: {user_question}'
        with st.spinner('Thinking...'):
            answer = ask_llama(prompt)
        st.markdown(f'**Agent:** {answer}')
    
