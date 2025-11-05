import requests
import gradio as gr

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "starcoder2:3b"

def chat_response(message: str, history: list) -> str:
    """Generate response using StarCoder2 3B"""
    
    context = ""
    for user_msg, bot_msg in history[-2:]:
        context += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
    
    prompt = f"""{context}User: {message}
Assistant:"""
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
                "top_p": 0.9,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: Status {response.status_code}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Build interface
with gr.Blocks(title="StarCoder2 3B") as demo:
    gr.Markdown("# 💻 StarCoder2 3B Code Assistant\n**Model:** starcoder2:3b | **GPU:** RTX 3050")
    
    chatbot = gr.Chatbot(height=450, type="tuples")
    msg = gr.Textbox(label="Ask for code", placeholder="Type your coding question...", lines=2)
    
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
    demo.launch(server_name="0.0.0.0", server_port=7860)
