import requests
import gradio as gr

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:3.8b"

def chat_response(user_input: str, history: list) -> str:
    """Generate response using Phi 3.8B"""
    
    # Build chat context
    messages = ""
    for user_msg, bot_msg in history[-3:]:  # Last 3 exchanges
        messages += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    
    full_prompt = messages + f"User: {user_input}\nAssistant:"
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "Error connecting to model"
    except Exception as e:
        return f"Error: {str(e)}"

def update_chat(user_input: str, history):
    """Update chat with new response"""
    bot_response = chat_response(user_input, history)
    history.append([user_input, bot_response])
    return history, ""

# Build Gradio interface
with gr.Blocks(title="Phi 3.8B Chat") as demo:
    gr.Markdown("# 💬 Phi 3.8B Chatbot")
    gr.Markdown("Simple local chatbot with Phi 3.8B")
    
    chatbot = gr.Chatbot(label="Chat", height=400)
    msg = gr.Textbox(label="Your message", lines=2)
    clear = gr.Button("Clear")
    
    msg.submit(update_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
