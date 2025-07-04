import gradio as gr
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.quantization import QuantizationManager
# from modules.inference_manager import InferenceManager
from modules.model_manager import ModelManager
from modules.ui_components import UIComponents

class GGUFWebUI:
    """
    GGUF WebUI - Main Application
    A web interface for quantizing and running inference on GGUF models
    """
    def __init__(self):
        self.quant_manager = QuantizationManager()
        # self.inference_manager = InferenceManager()
        self.model_manager = ModelManager()
        self.ui = UIComponents()
        
    def create_interface(self):
        """Create the main Gradio interface"""
        
        with gr.Blocks(
            title="GGUF Model WebUI",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto;
            }
            .tab-nav {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>GGUF WebUI</h1>
                <p>Convert Hugging Face models to GGUF format and run inference</p>
            </div>
            """)
            
            with gr.Tabs():
                # Model Conversion Tab
                with gr.Tab("🔄 Model Conversion", id="conversion"):
                    self.create_conversion_tab()
                
                # Model Inference Tab
                with gr.Tab("💬 Model Inference", id="inference"):
                    self.create_inference_tab()
                
                # Model Management Tab
                with gr.Tab("📁 Model Management", id="management"):
                    self.create_management_tab()
                
        return interface
    
    def create_conversion_tab(self):
        """Create the model conversion tab"""
        
        gr.HTML("<h3>Convert Hugging Face Models to GGUF</h3>")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model Input Section
                with gr.Group():
                    gr.HTML("<h4>📥 Model Input</h4>")
                    model_name = gr.Textbox(
                        label="Hugging Face Model Name",
                        placeholder="e.g., microsoft/DialoGPT-medium, Qwen/Qwen1.5-1.8B",
                        info="Enter the Hugging Face model repository name"
                    )
                    
                # Quantization Options
                with gr.Group():
                    gr.HTML("<h4>⚙️ Quantization Options</h4>")
                    quant_methods = gr.CheckboxGroup(
                        choices=[
                            ("q2_k - 2-bit quantization", "q2_k"),
                            ("q3_k_s - 3-bit quantization (Small)", "q3_k_s"),
                            ("q3_k_m - 3-bit quantization (Medium)", "q3_k_m"),
                            ("q3_k_l - 3-bit quantization (Large)", "q3_k_l"),
                            ("iq4_xs - 4-bit quantization (Extra Small)", "iq4_xs"),
                            ("q4_k_s - 4-bit quantization (Small)", "q4_k_s"),
                            ("q4_k_m - 4-bit quantization (Medium)", "q4_k_m"),
                            ("q5_k_s - 5-bit quantization (Small)", "q5_k_s"),
                            ("q5_k_m - 5-bit quantization (Medium)", "q5_k_m"),
                            ("q6_k - 6-bit quantization", "q6_k"),
                            ("q8_0 - 8-bit quantization", "q8_0"),
                            ("f16 - 16-bit float", "f16")
                        ],
                        value=["q4_k_m"],
                        label="Select Quantization Methods"
                    )
                    
                    select_all_btn = gr.Button("Select All Methods", size="sm")
                    select_recommended_btn = gr.Button("Select Recommended (q4_k_m, q5_k_m, q8_0)", size="sm")
                
                # Hugging Face Upload Options
                with gr.Group():
                    gr.HTML("<h4>🤗 Hugging Face Upload</h4>")
                    upload_to_hf = gr.Checkbox(label="Upload to Hugging Face Hub", value=False)
                    
                    with gr.Group(visible=False) as hf_options:
                        hf_token = gr.Textbox(
                            label="Hugging Face Token",
                            placeholder="Your HF token (will be hidden)",
                            type="password"
                        )
                        hf_repo_id = gr.Textbox(
                            label="Repository ID",
                            placeholder="username/model-name-gguf"
                        )
                        private_repo = gr.Checkbox(label="Private Repository", value=False)
                
                # Convert Button
                convert_btn = gr.Button("🚀 Start Conversion", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Status and Progress
                with gr.Group():
                    gr.HTML("<h4>📊 Status</h4>")
                    status_text = gr.Textbox(
                        label="Current Status",
                        value="Ready to convert",
                        interactive=False
                    )
                    progress_bar = gr.Progress()
                    
                # Conversion Log
                with gr.Group():
                    gr.HTML("<h4>📝 Conversion Log</h4>")
                    log_output = gr.Textbox(
                        label="Logs",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        show_copy_button=True
                    )
                
                # Download Links
                with gr.Group():
                    gr.HTML("<h4>⬇️ Download Results</h4>")
                    download_files = gr.File(
                        label="Download Quantized Models",
                        file_count="multiple",
                        interactive=False
                    )
        
        # Event handlers
        upload_to_hf.change(
            lambda x: gr.update(visible=x),
            inputs=[upload_to_hf],
            outputs=[hf_options]
        )
        
        select_all_btn.click(
            lambda: [method[1] for method in quant_methods.choices],
            outputs=[quant_methods]
        )
        
        select_recommended_btn.click(
            lambda: ["q4_k_m", "q5_k_m", "q8_0"],
            outputs=[quant_methods]
        )
        
        convert_btn.click(
            self.quant_manager.convert_model,
            inputs=[
                model_name, quant_methods, upload_to_hf, 
                hf_token, hf_repo_id, private_repo
            ],
            outputs=[status_text, log_output, download_files]
        )
    
    def create_inference_tab(self):
        """Create the model inference tab"""
        
        gr.HTML("<h3>Chat with GGUF Models</h3>")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model Selection
                with gr.Group():
                    gr.HTML("<h4>🎯 Model Selection</h4>")
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        info="Select a quantized model for inference"
                    )
                    refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
                    
                    # Model upload option
                    uploaded_model = gr.File(
                        label="Or Upload GGUF Model",
                        file_types=[".gguf"]
                    )
                
                # Generation Parameters
                with gr.Group():
                    gr.HTML("<h4>⚙️ Generation Parameters</h4>")
                    max_tokens = gr.Slider(
                        minimum=1, maximum=2048, value=512,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.9, step=0.1,
                        label="Top-p (nucleus sampling)"
                    )
                    repeat_penalty = gr.Slider(
                        minimum=1.0, maximum=1.5, value=1.1, step=0.1,
                        label="Repeat Penalty"
                    )
                
                # Model Info
                with gr.Group():
                    gr.HTML("<h4>ℹ️ Model Information</h4>")
                    model_info = gr.Textbox(
                        label="Model Details",
                        lines=5,
                        interactive=False
                    )
            
            with gr.Column(scale=2):
                # Chat Interface
                with gr.Group():
                    gr.HTML("<h4>💬 Chat Interface</h4>")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        export_btn = gr.Button("📤 Export Chat", size="sm")
                
                # Quick Prompts
                with gr.Group():
                    gr.HTML("<h4>🚀 Quick Prompts</h4>")
                    with gr.Row():
                        prompt_buttons = [
                            gr.Button("👋 Hello", size="sm"),
                            gr.Button("📝 Write a story", size="sm"),
                            gr.Button("🧮 Math problem", size="sm"),
                            gr.Button("💻 Code help", size="sm"),
                        ]
        
        # Event handlers for inference tab
        refresh_models_btn.click(
            self.model_manager.get_available_models,
            outputs=[model_dropdown]
        )
        
        # model_dropdown.change(
        #     self.inference_manager.load_model,
        #     inputs=[model_dropdown],
        #     outputs=[model_info]
        # )
        
        # uploaded_model.change(
        #     self.inference_manager.load_uploaded_model,
        #     inputs=[uploaded_model],
        #     outputs=[model_info]
        # )
        
        # send_btn.click(
        #     self.inference_manager.chat,
        #     inputs=[
        #         msg, chatbot, model_dropdown, max_tokens, 
        #         temperature, top_p, repeat_penalty
        #     ],
        #     outputs=[msg, chatbot]
        # )
        
        # msg.submit(
        #     self.inference_manager.chat,
        #     inputs=[
        #         msg, chatbot, model_dropdown, max_tokens, 
        #         temperature, top_p, repeat_penalty
        #     ],
        #     outputs=[msg, chatbot]
        # )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        )
        
        # Quick prompt handlers
        prompt_buttons[0].click(lambda: "Hello! How are you today?", outputs=[msg])
        prompt_buttons[1].click(lambda: "Write a short creative story about a robot learning to paint.", outputs=[msg])
        prompt_buttons[2].click(lambda: "Solve this math problem: What is 15% of 240?", outputs=[msg])
        prompt_buttons[3].click(lambda: "Help me write a Python function to calculate fibonacci numbers.", outputs=[msg])
    
    def create_management_tab(self):
        """Create the model management tab"""
        
        gr.HTML("<h3>Manage Your Models</h3>")
        
        with gr.Row():
            with gr.Column():
                # Model List
                with gr.Group():
                    gr.HTML("<h4>📋 Available Models</h4>")
                    model_list = gr.Dataframe(
                        headers=["Model Name", "Size", "Format", "Date Created"],
                        datatype=["str", "str", "str", "str"],
                        label="Your Models"
                    )
                    refresh_list_btn = gr.Button("🔄 Refresh List", size="sm")
                
                # Model Actions
                with gr.Group():
                    gr.HTML("<h4>🛠️ Model Actions</h4>")
                    selected_model = gr.Dropdown(
                        label="Select Model for Actions",
                        choices=[]
                    )
                    
                    with gr.Row():
                        delete_btn = gr.Button("🗑️ Delete", variant="secondary", size="sm")
                        info_btn = gr.Button("ℹ️ Info", size="sm")
                        benchmark_btn = gr.Button("📊 Benchmark", size="sm")
            
            with gr.Column():
                # Model Details
                with gr.Group():
                    gr.HTML("<h4>📊 Model Details</h4>")
                    model_details = gr.JSON(label="Model Information")
                
                # Storage Info
                with gr.Group():
                    gr.HTML("<h4>💾 Storage Information</h4>")
                    storage_info = gr.Textbox(
                        label="Storage Usage",
                        lines=5,
                        interactive=False
                    )
        
        # Event handlers for management tab
        refresh_list_btn.click(
            self.model_manager.get_model_list,
            outputs=[model_list, selected_model]
        )
        
        info_btn.click(
            self.model_manager.get_model_info,
            inputs=[selected_model],
            outputs=[model_details]
        )
        
        delete_btn.click(
            self.model_manager.delete_model,
            inputs=[selected_model],
            outputs=[model_list, selected_model]
        )
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Set default launch parameters
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "show_error": True,
            **kwargs
        }
        
        return interface.launch(**launch_params)

def main():
    """Main entry point"""
    app = GGUFWebUI()
    app.launch()

if __name__ == "__main__":
    main()