import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:9090")

def inference(message, history):
    partial_message = ""
    for token in client.text_generation(message, max_new_tokens=200, stream=True):
        partial_message += token
        yield partial_message

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="This is the demo for Gradio UI consuming TGI endpoint with LLaMA 7B-Chat model.",
    title="Gradio_TGI",
    examples=["Are tomatoes vegetables?", "Summarise Think and Grow Rich", "Create a javascript countdown to December 31st 2023", "Write a slogan for a coffee bean company",
"Create a 6 month employment contract for a freelance graphic designer",
"List 10 ideas for a million dollar product in 2023"], 

    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).queue().launch(share=True)
