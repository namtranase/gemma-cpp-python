import time
import streamlit as st
from pygemma import Gemma

st.set_page_config(page_title="Gemma 💬")
st.title("Gemma Cpp Python Chat Demo 🎈")

# Initialize session state for the model and its load status
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
    st.session_state["gemma"] = None
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How may I help you?"}
    ]


@st.cache_resource
def load_gemma_model(tokenizer_path, weights_path, model_type):
    gemma = Gemma()
    gemma.load_model(tokenizer_path, weights_path, model_type)
    return gemma


# Sidebar for model configuration
with st.sidebar:
    st.title("Gemma Config")
    tokenizer_path = st.text_input(
        "Tokenizer path", value="", placeholder="tokenizer.spm"
    )
    weights_path = st.text_input(
        "Compressed weights path", value="", placeholder="2b-it-sfp.sbs"
    )
    model_type = st.text_input("Model type", value="2b-it", placeholder="2b-it")

    # Load model button in the sidebar
    if st.button("Load Model"):
        st.session_state["gemma"] = load_gemma_model(
            tokenizer_path, weights_path, model_type
        )
        st.session_state["model_loaded"] = True

    # Indicate whether the model is loaded
    if st.session_state["model_loaded"]:
        st.sidebar.success("Model Loaded Successfully!")
    else:
        st.sidebar.warning('Model Not Loaded. Click "Load Model" to load the model.')

    st.markdown(
        "📖 Check the detail at [gemma-cpp-python](https://github.com/namtranase/gemma-cpp-python)!"
    )

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}
    ]

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    # Hugging Face Login
    if st.session_state.model_loaded:
        return st.session_state.gemma.completion(prompt_input)
    else:
        return "Please load the model first."


# User-provided prompt
if prompt := st.chat_input(disabled=not (st.session_state["model_loaded"])):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
