import streamlit as st
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from unsloth import FastLanguageModel
import torch

st.title("🎭 IMDB Sentiment Classifier — LoRA + Unsloth + Gemma 2B")

# Load model
@st.cache_resource
def load_model():
    adapter_path = "./unsloth_gemma2b_lora_adapter"
    config = PeftConfig.from_pretrained(adapter_path)
    
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_name_or_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Input
review = st.text_area("Enter a movie review 👇")
if st.button("Classify Sentiment"):
    with st.spinner("Classifying..."):
        prompt = f"Classify the sentiment of this review:\n\n{review}\nSentiment:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=5)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "positive" in response.lower():
            st.success("🔵 Sentiment: Positive")
        elif "negative" in response.lower():
            st.error("🔴 Sentiment: Negative")
        else:
            st.warning("⚠️ Sentiment unclear — try a longer review.")
