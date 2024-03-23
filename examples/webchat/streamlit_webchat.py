import time
import streamlit as st
from pygemma import Gemma
from process_web import WebProcesser


st.set_page_config(page_title="Chat with Website 💬")
st.title("Gemma Cpp Python Chat with Website Demo 🎈")

# Initialize session state for the model and its load status
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
    st.session_state["website_loaded"] = False
    st.session_state["gemma"] = None
    st.session_state["web_processer"] = None
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How may I help you?"}
    ]


@st.cache_resource
def load_gemma_model(tokenizer_path, weights_path, model_type):
    gemma = Gemma()
    gemma.load_model(tokenizer_path, weights_path, model_type)
    return gemma


@st.cache_resource
def load_web_processer():
    web_processor = WebProcesser()
    return web_processor


# Sidebar for model configuration
with st.sidebar:
    st.title("Gemma Config")
    tokenizer_path = st.text_input(
        "Tokenizer path", value="tokenizer.spm", placeholder="tokenizer.spm"
    )
    weights_path = st.text_input(
        "Compressed weights path", value="2b-it-sfp.sbs", placeholder="2b-it-sfp.sbs"
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

    st.markdown("## Website Processing")
    website_url = st.text_input(
        "Website URL",
        value="https://namtranase.github.io/terminalmind/",
        placeholder="Enter website URL",
    )

    if st.button("Process Website"):
        st.session_state["web_processor"] = WebProcesser()
        if website_url:
            # Placeholder for the function to process website data
            st.session_state["web_processor"].init_db_website(website_url)
            st.session_state["website_loaded"] = True
            st.sidebar.success("Website processed successfully, Now you can ask!")
        else:
            st.sidebar.error("Please enter a valid website URL.")

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
    if st.session_state.model_loaded and st.session_state.website_loaded:
        context = st.session_state["web_processor"].get_context(prompt_input)
        prompt = f"""Use the following pieces of context to answer the question at the end.
        Context: {context}.\n
        Question: {prompt_input}
        Helpful Answer:"""
        return st.session_state.gemma.completion(prompt)
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
