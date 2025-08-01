import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Ensure .streamlit directory exists to avoid FileNotFoundError
os.makedirs(os.path.expanduser("~/.streamlit"), exist_ok=True)


# Load the model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dialogpt-finetuned/final")
    model = AutoModelForCausalLM.from_pretrained("dialogpt-finetuned/final", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "chat_history_text" not in st.session_state:
    st.session_state.chat_history_text = ""

st.title("🎬 Movie Chatbot (DialoGPT)")
user_input = st.text_input("You:")

if user_input:
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    output_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.write(f"**Bot**: {response}")

    st.session_state.chat_history_ids = bot_input_ids
    st.session_state.chat_history_text += f"You: {user_input}\nBot: {response}\n"

if st.session_state.chat_history_text:
    st.text_area("Conversation so far:", st.session_state.chat_history_text, height=300)
