import requests
import gradio as gr

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:3.8b"

def chat_response(message: str, history: list) -> str:
    """Generate response using Phi 3.8B"""
    
    # Build simple conversation context
    context = ""
    for user_msg, bot_msg in history[-3:]:  # Last 3 exchanges
        context += f"User: {user_msg}\nBot: {bot_msg}\n"
    
    prompt = context + f"User: {message}\nBot:"
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: Could not get response (status {response.status_code})"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Simple Gradio interface
with gr.Blocks(title="Phi 3.8B Chatbot") as demo:
    gr.Markdown("# 🤖 Phi 3.8B Local Chatbot\n**Model:** phi3:3.8b | **GPU:** RTX 3050")
    
    chatbot = gr.Chatbot(height=400, type="tuples")
    msg = gr.Textbox(label="Your message", placeholder="Type here...", lines=2)
    
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    def respond(message, chat_history):
        bot_message = chat_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
