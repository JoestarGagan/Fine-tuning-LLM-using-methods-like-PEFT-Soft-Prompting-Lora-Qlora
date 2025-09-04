# Sentiment Analysis with Gemma-2B + Unsloth (IMDb Reviews)


This project essentially fine-tunes Gemma-2B (Google’s lightweight open LLM) on the IMDb movie review dataset using parameter-efficient fine-tuning (PEFT) with Unsloth.
The fine-tuned model can classify reviews as positive or negative (sentiment analysis), and we deploy it via a Streamlit app for easy interaction.


# Project Overview :-

Model: Gemma-2B (quantized with 4-bit for efficiency)
Frameworks: Hugging Face 🤗 Transformers, Unsloth, PyTorch
Techniques: LoRA (PEFT), soft prompting, layer dropping
Dataset: IMDb reviews (20k samples, balanced sentiment labels)
Deployment: Streamlit frontend with saved .h5 model


# Repository Structure :-

├── app.py             # Streamlit app for deployment

├── genai_project.py   # Fine-tuning Code

├── unsloth_gemma2b_lora_adapter-1epoch.zip    # Fine-tuned model weights (exported after training)

└── README.md          # Project documentation


# Environment Setup :-
Install required libraries

```py
!pip install torch transformers datasets evaluate accelerate
!pip install unsloth
!pip install streamlit
```

# Training the Model

Run the training script:

```py
python genai_project.py
```

What happens inside:

- Preprocessing – IMDb data is cleaned and split into train/validation.

- Tokenization – Reviews are tokenized using Gemma’s tokenizer.

- Fine-tuning :

  - Uses QLoRA + PEFT for efficiency

  - Layer dropping for regularization

- Evaluation – Accuracy & F1-score tracked during training.

- Saving – Model adapters saved as sentiment_model.h5.


# Deployment with Streamlit

Run the app locally:
```py
streamlit run app.py
```

Workflow:

- Loads the .h5 fine-tuned weights

- User enters a movie review in the text box

- Model predicts Positive or Negative sentiment

- Instant feedback via Streamlit UI



