import gradio as gr
import os
import sys
from pathlib import Path
import threading
import time
import asyncio
from typing import List, Tuple
import queue

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from modules.quantization import QuantizationManager
    from modules.model_manager import ModelManager
    from modules.ui_components import UIComponents
    from modules.inference import InferenceManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory")
    sys.exit(1)

class GGUFWebUI:
    """
    GGUF WebUI - Main Application
    A web interface for quantizing and running inference on GGUF models
    """
    
    def __init__(self):
        self.load_initial_settings()
    
    # Initialize managers with loaded settings
        base_dir = getattr(self, 'config_base_dir', './models')
        self.model_manager = ModelManager(base_dir)
        self.quant_manager = QuantizationManager()
        self.ui = UIComponents()
        self.inference_manager = InferenceManager()
        
        # Status tracking
        self.current_operation = None
        self.operation_thread = None
        self.log_queue = queue.Queue()
        self.stop_log_streaming = False
    



    def load_initial_settings(self):
        """Load initial settings from config file"""
        try:
            config = self.load_settings_from_config()
            self.config_base_dir = config.get('base_dir', './models')
            print(f"Loaded base directory from config: {self.config_base_dir}")
        except Exception as e:
            print(f"No config found, using defaults: {e}")
            self.config_base_dir = './models'


    def create_interface(self):
        """Create the main Gradio interface"""
        with gr.Blocks(
            title="GGUF Model WebUI",
            theme=self.ui.theme,
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: auto;
            }
            .status-box {
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                padding: 10px;
                margin: 10px 0;
            }
            .success-status {
                border-color: #28a745;
                background-color: #d4edda;
            }
            .error-status {
                border-color: #dc3545;
                background-color: #f8d7da;
            }
            .processing-status {
                border-color: #007bff;
                background-color: #d1ecf1;
            }
            .log-container {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #2c3e50; margin-bottom: 10px;">🔧 GGUF Model WebUI</h1>
                <p style="color: #7f8c8d; font-size: 18px;">Convert Hugging Face models to GGUF format and run inference</p>
            </div>
            """)
            
            # Create tabs
            with gr.Tabs():
                # Quantization Tab
                with gr.TabItem("🔄 Model Quantization", id="quantization"):
                    self.create_quantization_tab(interface)
                
                # Inference Tab
                with gr.TabItem("🤖 Model Inference", id="inference"):
                    self.create_inference_tab(interface)
                
                # Model Management Tab
                with gr.TabItem("📁 Model Management", id="management"):
                    self.create_management_tab(interface)
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings", id="settings"):
                    self.create_settings_tab(interface)
        
        return interface
    
    def create_quantization_tab(self, interface):
        """Create the quantization tab with real-time updates"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>📥 Model Input</h3>")
                
                # Model input
                model_input = gr.Textbox(
                    label="Hugging Face Model Name",
                    placeholder="e.g., microsoft/DialoGPT-medium, Qwen/Qwen1.5-1.8B",
                    info="Enter the Hugging Face model repository name",
                    lines=1
                )
                
                # Quantization methods
                quant_methods = gr.CheckboxGroup(
                    label="Quantization Methods",
                    choices=self.quant_manager.get_available_methods(),
                    value=["q4_k_m"],
                    info="Select one or more quantization methods"
                )
                
                # Advanced options
                with gr.Accordion("🔧 Advanced Options", open=False):
                    with gr.Row():
                        with gr.Column():
                            base_model_dir = gr.Textbox(
                                label="Base Model Directory",
                                value="./models/original",
                                info="Directory to store downloaded models"
                            )
                            
                            quantized_dir = gr.Textbox(
                                label="Quantized Models Directory", 
                                value="./models/quantized",
                                info="Directory to store quantized models"
                            )
                
                # Hugging Face Upload Options
                with gr.Accordion("🤗 Hugging Face Upload", open=False):
                    upload_to_hf = gr.Checkbox(
                        label="Upload to Hugging Face Hub",
                        value=False,
                        info="Upload quantized models to Hugging Face"
                    )
                    
                    with gr.Row():
                        hf_repo_id = gr.Textbox(
                            label="Repository ID",
                            placeholder="username/model-name-gguf",
                            info="Hugging Face repository ID"
                        )
                        
                        hf_token = gr.Textbox(
                            label="HF Token",
                            placeholder="hf_...",
                            type="password",
                            info="Hugging Face API token"
                        )
                    
                    private_repo = gr.Checkbox(
                        label="Private Repository",
                        value=False,
                        info="Make repository private"
                    )
                
                # Action buttons
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Quantization", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop Process", variant="stop", size="lg")
                    clear_btn = gr.Button("🧹 Clear Logs", variant="secondary", size="lg")
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Process Status</h3>")
                
                # Status display
                status_display = gr.Textbox(
                    label="Current Status",
                    value="Ready",
                    interactive=False,
                    elem_classes="status-box"
                )
                
                # Log display with auto-scroll
                log_display = gr.Textbox(
                    label="Process Log",
                    value="",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes="status-box log-container",
                    autoscroll=True,
                )
                
                # Output files
                output_files = gr.File(
                    label="Generated Files",
                    file_count="multiple",
                    interactive=False,
                    visible=False
                )
        
        # Custom log streaming function
        def get_streaming_logs():
            """Get all logs including real-time updates"""
            return self.quant_manager.get_log_text()

        def get_current_status():
            """Get current status"""
            return self.quant_manager.get_status()
        
        # Modified quantization manager to use queue
        def setup_log_streaming():
            """Setup log streaming - no changes needed, QuantizationManager handles it"""
            pass  # The QuantizationManager already handles logging properly
        
        # Event handlers
        def start_quantization(model_name, methods, upload_hf, token, repo_id, private, 
                             base_dir, quant_dir, progress=gr.Progress()):
            """Start the quantization process"""
            if not model_name.strip():
                return "❌ Please enter a model name", "", gr.update(visible=False)
            
            if not methods:
                return "❌ Please select at least one quantization method", "", gr.update(visible=False)
            
            # Setup log streaming
            setup_log_streaming()
            self.stop_log_streaming = False
            
            # Clear previous logs
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break
            
            def run_conversion():
                try:
                    progress(0, desc="Starting quantization...")
                    
                    # Add progress tracking
                    def progress_callback(step, total, description):
                        progress(step / total, desc=description)
                    
                    # If your quantization manager supports progress callback
                    result = self.quant_manager.convert_model(
                        model_name=model_name,
                        quant_methods=methods,
                        upload_to_hf=upload_hf,
                        hf_token=token,
                        hf_repo_id=repo_id,
                        private_repo=private,
                        base_model_dir=base_dir if base_dir.strip() else None,
                        quantized_dir=quant_dir if quant_dir.strip() else None
                        # progress_callback=progress_callback  # Add this if supported
                    )
                    
                    progress(1.0, desc="Quantization completed!")
                    return result
                except Exception as e:
                    error_msg = f"❌ Conversion error: {str(e)}"
                    self.quant_manager.log_message(error_msg, "ERROR")
                    progress(0, desc="Error occurred")
                    return error_msg
            
            # Start the process in a thread
            if not self.quant_manager.is_busy():
                self.operation_thread = threading.Thread(target=run_conversion)
                self.operation_thread.daemon = True
                self.operation_thread.start()
                return "🚀 Quantization started...", "Process initiated", gr.update(visible=False)
            else:
                return "⚠️ Another process is already running", "", gr.update(visible=False)
        
        def stop_process():
            """Stop the current process"""
            self.stop_log_streaming = True
            if self.quant_manager.is_busy():
                self.quant_manager.stop_process()
                return "⏹️ Stop signal sent", "Process stop requested"
            else:
                return "ℹ️ No active process to stop", ""
        
        def clear_logs():
            """Clear the logs"""
            # Clear queue
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.quant_manager.clear_log()
            return ""
        
        def update_status():
            """Update status only"""
            return self.quant_manager.get_status()
        
        # Real-time log streaming
        def get_streaming_logs():
            """Get logs for streaming"""
            return self.quant_manager.get_log_text()
        
        # Connect events
        start_btn.click(
            fn=start_quantization,
            inputs=[
                model_input, quant_methods, upload_to_hf, 
                hf_token, hf_repo_id, private_repo,
                base_model_dir, quantized_dir
            ],
            outputs=[status_display, log_display, output_files]
        )
        
        stop_btn.click(
            fn=stop_process,
            outputs=[status_display, log_display]
        )
        
        clear_btn.click(
            fn=clear_logs,
            outputs=[log_display]
        )
        
        timer = gr.Timer(2.0)  # Update every 2 seconds
    
        def update_logs_and_status():
            """Update both logs and status"""
            try:
                logs = self.quant_manager.get_log_text()
                status = self.quant_manager.get_status()
                return logs, status
            except Exception as e:
                return f"Error updating display: {e}", "Error"
        
        # Connect timer to update function
        timer.tick(
            fn=update_logs_and_status,
            outputs=[log_display, status_display]
        )
    
    def create_inference_tab(self, interface):
        """Create the inference tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🤖 Model Inference</h3>")
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=[],
                    info="Choose a quantized model for inference"
                )
                
                # Model upload option
                model_upload = gr.File(
                    label="Or Upload GGUF Model",
                    file_types=[".gguf"],
                    file_count="single"
                )
                
                # Load model button
                load_model_btn = gr.Button("📂 Load Model", variant="primary")
                
                # Model info display
                model_info = gr.Textbox(
                    label="Model Information",
                    lines=6,
                    interactive=False
                )
                
                # Generation parameters
                with gr.Accordion("⚙️ Generation Parameters", open=False):
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=1,
                        maximum=2048,
                        value=512,
                        step=1
                    )
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1
                    )
                    
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1
                    )
                    
                    repeat_penalty = gr.Slider(
                        label="Repeat Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1
                    )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>💬 Chat Interface</h3>")
                
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    export_chat_btn = gr.Button("📤 Export Chat", variant="secondary")
                    benchmark_btn = gr.Button("⚡ Benchmark", variant="secondary")
        
        # Inference event handlers
        def refresh_model_list():
            models = self.inference_manager.get_available_models()
            return gr.update(choices=models)
        
        def load_selected_model(model_path, uploaded_file):
            if uploaded_file:
                result = self.inference_manager.load_uploaded_model(uploaded_file)
            elif model_path:
                result = self.inference_manager.load_model(model_path)
            else:
                result = "Please select a model or upload a file"
            return result
        
        def chat_fn(message, history, model_path, max_tok, temp, top_p_val, repeat_pen):
            return self.inference_manager.chat(
                message, history, model_path, max_tok, temp, top_p_val, repeat_pen
            )
        
        def clear_chat():
            return [], ""
        
        def export_chat_fn(history):
            return self.inference_manager.export_chat(history)
        
        def benchmark_fn(model_path):
            return self.inference_manager.benchmark_model(model_path)
        
        # Connect inference events
        interface.load(refresh_model_list, outputs=[model_dropdown])
        
        load_model_btn.click(
            fn=load_selected_model,
            inputs=[model_dropdown, model_upload],
            outputs=[model_info]
        )
        
        send_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot, model_dropdown, max_tokens, temperature, top_p, repeat_penalty],
            outputs=[msg_input, chatbot]
        )
        
        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot, model_dropdown, max_tokens, temperature, top_p, repeat_penalty],
            outputs=[msg_input, chatbot]
        )
        
        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input]
        )
        
        export_chat_btn.click(
            fn=export_chat_fn,
            inputs=[chatbot],
            outputs=[model_info]
        )
        
        benchmark_btn.click(
            fn=benchmark_fn,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
    
    def create_management_tab(self, interface):
        """Create the model management tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>📋 Model Library</h3>")
                
                # Model list table
                model_table = gr.DataFrame(
                    label="Available Models",
                    headers=["Name", "Size", "Type", "Created"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True
                )
                
                # Model actions
                with gr.Row():
                    selected_model = gr.Dropdown(
                        label="Select Model",
                        choices=[],
                        interactive=True,
                        info="Choose a model to manage"
                    )
                    
                    refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                    delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                
                # Model upload
                with gr.Accordion("📤 Upload Model", open=False):
                    uploaded_model = gr.File(
                        label="Upload GGUF Model",
                        file_types=[".gguf"],
                        file_count="single"
                    )
                    
                    model_name = gr.Textbox(
                        label="Model Name (Optional)",
                        placeholder="my-custom-model",
                        info="Leave empty to use original filename"
                    )
                    
                    model_description = gr.Textbox(
                        label="Description",
                        placeholder="Description of the model...",
                        lines=3
                    )
                    
                    upload_btn = gr.Button("📁 Import Model", variant="primary")
            
            with gr.Column(scale=1):
                gr.HTML("<h3>ℹ️ Model Information</h3>")
                
                # Model details
                model_info = gr.Textbox(
                    label="Model Details",
                    lines=12,
                    interactive=False,
                    show_copy_button=True
                )
                
                # Storage info
                storage_info = gr.Textbox(
                    label="Storage Information",
                    lines=10,
                    interactive=False
                )
                
                # Management actions
                with gr.Column():
                    export_btn = gr.Button("📊 Export Model List", variant="secondary")
                    cleanup_btn = gr.Button("🧹 Cleanup Old Models", variant="secondary")
        
        # Management event handlers
        def refresh_models():
            table_data, choices = self.model_manager.get_available_models()
            storage = self.model_manager.get_storage_info()
            return gr.update(value=table_data), gr.update(choices=choices), gr.update(value=storage)
        
        def delete_model(model_path):
            if model_path:
                table_data, choices = self.model_manager.delete_model(model_path)
                return gr.update(value=table_data), gr.update(choices=choices), "Model deleted"
            return gr.update(), gr.update(), "No model selected"
        
        def show_model_info(model_path):
            if model_path:
                info = self.model_manager.get_model_detailed_info(model_path)
                return str(info)
            return ""
        
        def import_model(file, name, description):
            if file:
                result = self.model_manager.import_model(file.name, name, description)
                table_data, choices = self.model_manager.get_available_models()
                return result, gr.update(value=table_data), gr.update(choices=choices)
            return "No file selected", gr.update(), gr.update()
        
        # Connect management events
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model_table, selected_model, storage_info]
        )
        
        delete_btn.click(
            fn=delete_model,
            inputs=[selected_model],
            outputs=[model_table, selected_model, model_info]
        )
        
        selected_model.change(
            fn=show_model_info,
            inputs=[selected_model],
            outputs=[model_info]
        )
        
        upload_btn.click(
            fn=import_model,
            inputs=[uploaded_model, model_name, model_description],
            outputs=[model_info, model_table, selected_model]
        )
        
        export_btn.click(
            fn=lambda: self.model_manager.export_model_list(),
            outputs=[model_info]
        )
        
        cleanup_btn.click(
            fn=lambda: self.model_manager.cleanup_old_models(),
            outputs=[model_info]
        )
        
        # Load initial data
        interface.load(refresh_models, outputs=[model_table, selected_model, storage_info])
    
    def create_settings_tab(self, interface):
        """Create settings tab with vertical layout"""
        
        # Top section - Directory Settings
        with gr.Group():
            gr.HTML("<h3>📁 Model Directories</h3>")
            
            with gr.Row():
                with gr.Column(scale=3):
                    base_dir_input = gr.Textbox(
                        label="Base Model Directory",
                        value=str(self.model_manager.base_dir),
                        placeholder="./models",
                        info="Base directory for all model operations",
                        show_copy_button=True
                    )
                
                with gr.Column(scale=1):
                    browse_base_btn = gr.Button("📁 Browse", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=3):
                    quantized_dir_input = gr.Textbox(
                        label="Quantized Models Directory",
                        value=str(self.model_manager.quantized_models_dir),
                        placeholder="./models/quantized",
                        info="Directory for quantized models",
                        show_copy_button=True
                    )
                
                with gr.Column(scale=1):
                    browse_quantized_btn = gr.Button("📁 Browse", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=3):
                    uploaded_dir_input = gr.Textbox(
                        label="Uploaded Models Directory",
                        value=str(self.model_manager.uploaded_models_dir),
                        placeholder="./models/uploaded",
                        info="Directory for uploaded models",
                        show_copy_button=True
                    )
                
                with gr.Column(scale=1):
                    browse_uploaded_btn = gr.Button("📁 Browse", variant="secondary")
        
        # Status section
        with gr.Group():
            gr.HTML("<h3>📊 Status Information</h3>")
            
            with gr.Row():
                    
                
                with gr.Column(scale=1):
                    
                    deps_status = gr.Textbox(
                        label="Dependencies Status",
                        value="Click 'Test Dependencies' to check system requirements",
                        lines=6,
                        interactive=False,
                        show_copy_button=True
                    )
                
                # Action buttons
                with gr.Row():
                    apply_btn = gr.Button("💾 Apply Settings", variant="primary")
                    reset_btn = gr.Button("🔄 Reset to Default", variant="secondary")
                    test_dirs_btn = gr.Button("📁 Test Directories", variant="stop")
                    test_deps_btn = gr.Button("🧪 Test Dependencies", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                # Status displays
                settings_status = gr.Textbox(
                    label="Settings Status",
                    value="Ready to apply settings",
                    interactive=False,
                    lines=5
                )
                
                # Directory validation
                dir_validation = gr.Textbox(
                    label="Directory Validation",
                    value="Click 'Test Directories' to validate paths",
                    interactive=False,
                    lines=8
                )
                
                # Current configuration display
                current_config = gr.JSON(
                    label="Current Configuration",
                    value={}
                )
        
        def test_dependencies():
            """Test system dependencies"""
            try:
                import subprocess
                import sys
                import importlib
                
                results = []
                
                # Test Python version
                python_version = sys.version.split()[0]
                results.append(f"✅ Python: {python_version}")
                
                # Test required packages
                required_packages = [
                    'gradio', 'transformers', 'huggingface_hub',
                    'torch', 'numpy', 'pathlib'
                ]
                
                for package in required_packages:
                    try:
                        module = importlib.import_module(package)
                        version = getattr(module, '__version__', 'Unknown')
                        results.append(f"✅ {package}: {version}")
                    except ImportError:
                        results.append(f"❌ {package}: NOT INSTALLED")
                
                # Test directories
                test_dirs = [
                    ("Base", self.model_manager.base_dir),
                    ("Quantized", self.model_manager.quantized_models_dir),
                    ("Uploaded", self.model_manager.uploaded_models_dir)
                ]
                
                for name, directory in test_dirs:
                    if directory.exists() and directory.is_dir():
                        results.append(f"✅ {name} Directory: OK")
                    else:
                        results.append(f"❌ {name} Directory: Missing")
                
                return "\n".join(results)
                
            except Exception as e:
                return f"❌ Error testing dependencies: {str(e)}"
        
        def apply_settings(base_dir, quantized_dir, uploaded_dir):
            """Apply new directory settings with proper validation"""
            try:
                # Validate paths
                base_path = Path(base_dir).resolve()
                quantized_path = Path(quantized_dir).resolve()
                uploaded_path = Path(uploaded_dir).resolve()
                
                # Update model manager
                self.model_manager.base_dir = base_path
                self.model_manager.quantized_models_dir = quantized_path
                self.model_manager.uploaded_models_dir = uploaded_path
                
                # Create directories
                quantized_path.mkdir(parents=True, exist_ok=True)
                uploaded_path.mkdir(parents=True, exist_ok=True)
                
                # Save to config
                settings_dict = {
                    'base_dir': str(base_path),
                    'quantized_dir': str(quantized_path),
                    'uploaded_dir': str(uploaded_path),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.save_settings_to_config(settings_dict)
                
                # Update quantization manager
                if hasattr(self, 'quant_manager'):
                    self.quant_manager.base_model_dir = str(base_path)
                    self.quant_manager.quantized_dir = str(quantized_path)
                
                return f"✅ Settings applied successfully!\nDirectories updated and saved to config.json"
                
            except Exception as e:
                return f"❌ Error applying settings: {str(e)}"
        
        def test_directories(base_dir, quantized_dir, uploaded_dir):
            """Test and validate directory paths"""
            validation_results = []
            
            directories = {
                "Base": base_dir,
                "Quantized": quantized_dir,
                "Uploaded": uploaded_dir
            }
            
            for name, path in directories.items():
                try:
                    path_obj = Path(path).resolve()
                    
                    if path_obj.exists():
                        if path_obj.is_dir():
                            items = list(path_obj.iterdir())
                            gguf_files = list(path_obj.glob("**/*.gguf"))
                            validation_results.append(f"✅ {name}: EXISTS ({len(items)} items, {len(gguf_files)} .gguf files)")
                        else:
                            validation_results.append(f"❌ {name}: EXISTS but not a directory")
                    else:
                        if path_obj.parent.exists():
                            validation_results.append(f"⚠️ {name}: CAN CREATE")
                        else:
                            validation_results.append(f"❌ {name}: PARENT MISSING")
                            
                except Exception as e:
                    validation_results.append(f"❌ {name}: ERROR - {str(e)}")
            
            return "\n".join(validation_results)
        
        def load_current_config():
            """Load and display current configuration"""
            config = self.load_settings_from_config()
            return config
        
        # Connect events
        apply_btn.click(
            fn=apply_settings,
            inputs=[base_dir_input, quantized_dir_input, uploaded_dir_input],
            outputs=[settings_status]
        )
        
        test_deps_btn.click(
            fn=test_dependencies,
            outputs=[deps_status]
        )
        
        test_dirs_btn.click(
            fn=test_directories,
            inputs=[base_dir_input, quantized_dir_input, uploaded_dir_input],
            outputs=[dir_validation]
        )
        
        # Load current config on interface load
        interface.load(
            fn=load_current_config,
            outputs=[current_config]
        )

    def save_settings_to_config(self, settings):
        """Save settings to configuration file with proper error handling"""
        try:
            import json
            config_file = Path("config.json")
            config = {}
            
            # Load existing config if it exists and is valid
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only load if file has content
                            config = json.loads(content)
                except (json.JSONDecodeError, FileNotFoundError):
                    print("Config file corrupted or empty, creating new one")
                    config = {}
            
            # Update with new settings
            config.update(settings)
            
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save updated config with proper formatting
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"Settings saved successfully to {config_file}")
            print(f"Saved settings: {json.dumps(config, indent=2)}")
            
        except Exception as e:
            print(f"Error saving config: {e}")
            raise e

    def load_settings_from_config(self):
        import json
        """Load settings from configuration file"""
        try:
            config_file = Path("config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        config = json.loads(content)
                        return config
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
            
    def create_model_management_with_settings(self, interface):
        """Enhanced model management with settings integration"""
        
        # Settings section
        self.create_settings_tab(interface)
        
        # Model management section  
        with gr.Row():
            refresh_models_btn = gr.Button("🔄 Refresh Models", variant="primary")
            debug_scan_btn = gr.Button("🔍 Debug Scan", variant="secondary")
        
        model_list_display = gr.Dataframe(
            headers=["Model Name", "Size", "Type", "Path"],
            value=[],
            label="Available Models",
            interactive=False
        )
        
        debug_output = gr.Textbox(
            label="Debug Information",
            lines=15,
            interactive=False,
            visible=False
        )
        
        def refresh_and_scan():
            """Refresh model list after settings change"""
            try:
                # Force rescan
                models = self.model_manager.scan_models()
                
                # Prepare table data
                table_data = []
                for model in models:
                    table_data.append([
                        model["name"],
                        model["size_str"], 
                        model["type"],
                        model["path"]
                    ])
                
                return table_data, gr.update(visible=False)
            except Exception as e:
                return [], gr.update(visible=True, value=f"Error: {e}")
        
        def debug_scan():
            """Debug the scanning process"""
            debug_info = self.model_manager.debug_scan_models()
            return gr.update(visible=True, value=debug_info)
        
        # Connect events
        refresh_models_btn.click(
            fn=refresh_and_scan,
            outputs=[model_list_display, debug_output]
        )
        
        debug_scan_btn.click(
            fn=debug_scan,
            outputs=[debug_output]
        )
        
        # Auto-refresh on interface load
        interface.load(
            fn=refresh_and_scan,
            outputs=[model_list_display, debug_output]
        )
    
    def get_system_info(self):
        """Get system information"""
        try:
            import platform
            
            info = []
            info.append(f"🖥️ OS: {platform.system()} {platform.release()}")
            info.append(f"🔧 Python: {platform.python_version()}")
            
            # Try to get psutil info
            try:
                import psutil
                info.append(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
                info.append(f"💽 Disk: {psutil.disk_usage('/').total / (1024**3):.1f} GB")
                info.append(f"🔥 CPU Usage: {psutil.cpu_percent()}%")
                info.append(f"📊 Memory Usage: {psutil.virtual_memory().percent}%")
            except ImportError:
                info.append("⚠️ psutil not installed - install for detailed system info")
            
            # Check git
            try:
                import subprocess
                git_version = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=10)
                if git_version.returncode == 0:
                    info.append(f"📝 Git: {git_version.stdout.strip()}")
                else:
                    info.append("❌ Git: Not working properly")
            except:
                info.append("❌ Git: Not installed")
            
            # Check cmake
            try:
                cmake_version = subprocess.run(["cmake", "--version"], capture_output=True, text=True, timeout=10)
                if cmake_version.returncode == 0:
                    info.append(f"🔨 CMake: {cmake_version.stdout.split()[2]}")
                else:
                    info.append("❌ CMake: Not working properly")
            except:
                info.append("❌ CMake: Not installed")
            
            # Check CUDA if available
            try:
                cuda_version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
                if cuda_version.returncode == 0:
                    info.append("🚀 CUDA: Available")
                else:
                    info.append("ℹ️ CUDA: Not available (CPU only)")
            except:
                info.append("ℹ️ CUDA: Not available (CPU only)")
            
            # Quantization manager status
            info.append(f"⚙️ Quantization Status: {'Busy' if self.quant_manager.is_busy() else 'Ready'}")
            
            # Model counts
            models = self.model_manager.scan_models()
            info.append(f"📦 Available Models: {len(models)}")
            
            return "\n".join(info)
        except Exception as e:
            return f"Error getting system info: {str(e)}"
    
    def launch(self, share=False, debug=False, server_name="127.0.0.1", server_port=7860):
        """Launch the web interface"""
        interface = self.create_interface()
        
        # Launch with custom settings
        interface.launch(
            share=share,
            debug=debug,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False,
            inbrowser=True,
        )

def main():
    """Main function to run the web UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Model WebUI")
    parser.add_argument("--share", action="store_true", help="Share the interface publicly")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    
    args = parser.parse_args()
    
    # Create and launch the web UI
    try:
        webui = GGUFWebUI()
        webui.launch(
            share=args.share,
            debug=args.debug,
            server_name=args.host,
            server_port=8080
        )
    except Exception as e:
        print(f"Error launching WebUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()