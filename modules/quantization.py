import os
import sys
import subprocess
import shutil
import tempfile
import threading
import time
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr
from huggingface_hub import snapshot_download, HfApi, create_repo, login
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationManager:
    """
    Quantization Manager Module with Real-time Logging
    Handles model quantization operations for the GGUF WebUI
    """

    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Directory structure - these can be updated
        self.original_models_dir = self.base_dir / "original"
        self.quantized_models_dir = self.base_dir / "quantized"
        self.llama_cpp_dir = self.base_dir / "llama.cpp"
        
        # Create directories
        self.original_models_dir.mkdir(exist_ok=True)
        self.quantized_models_dir.mkdir(exist_ok=True)
        
        # Available quantization methods
        self.quant_methods = {
            "q2_k": "2-bit quantization",
            "q3_k_s": "3-bit quantization - Small",
            "q3_k_m": "3-bit quantization - Medium", 
            "q3_k_l": "3-bit quantization - Large",
            "iq4_xs": "4-bit quantization - Extra Small",
            "q4_k_s": "4-bit quantization - Small",
            "q4_k_m": "4-bit quantization - Medium",
            "q5_k_s": "5-bit quantization - Small",
            "q5_k_m": "5-bit quantization - Medium",
            "q6_k": "6-bit quantization",
            "q8_0": "8-bit quantization",
            "f16": "16-bit float"
        }
        
        # Status tracking with thread-safe queue
        self.current_status = "Ready"
        self.log_queue = queue.Queue()
        self.current_log = []
        self.is_processing = False
        self.current_process = None
        self.stop_event = threading.Event()

    def update_directories(self, base_model_dir: str = None, quantized_dir: str = None):
        """Update directory paths"""
        if base_model_dir:
            self.original_models_dir = Path(base_model_dir)
            self.original_models_dir.mkdir(exist_ok=True)
        
        if quantized_dir:
            self.quantized_models_dir = Path(quantized_dir)
            self.quantized_models_dir.mkdir(exist_ok=True)

    def log_message(self, message: str, level: str = "INFO"):
        """Add a message to the log with thread safety"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.current_log.append(log_entry)
        self.log_queue.put(log_entry)
        logger.info(message)

    def get_log_text(self) -> str:
        """Get the current log as text"""
        # Get any new messages from queue
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break
        return "\n".join(self.current_log)

    def clear_log(self):
        """Clear the current log"""
        self.current_log = []
        # Clear the queue
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

    def run_subprocess_with_logging(self, cmd: List[str], cwd: str = None) -> bool:
        """Run subprocess with real-time output logging"""
        try:
            self.log_message(f"Running command: {' '.join(cmd)}")
            
            # Start process with pipes for stdout/stderr
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                universal_newlines=True,
                bufsize=1,  # Line buffered
                cwd=cwd
            )
            
            self.current_process = process
            
            # Read output line by line in real-time
            while True:
                if self.stop_event.is_set():
                    process.terminate()
                    return False
                    
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    # Clean and log the output
                    clean_output = output.strip()
                    if clean_output:
                        self.log_message(clean_output)
            
            # Wait for process to complete
            return_code = process.poll()
            self.current_process = None
            
            if return_code == 0:
                self.log_message("✅ Command completed successfully")
                return True
            else:
                self.log_message(f"❌ Command failed with return code: {return_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Error running subprocess: {e}", "ERROR")
            return False

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            # Check Python packages
            import huggingface_hub
            self.log_message("✓ huggingface_hub is available")
            
            # Check if git is available
            if not shutil.which("git"):
                self.log_message("✗ Git is not installed", "ERROR")
                return False
            self.log_message("✓ Git is available")
            
            # Check if cmake is available
            if not shutil.which("cmake"):
                self.log_message("⚠ CMake is not installed - will try to install", "WARNING")
            else:
                self.log_message("✓ CMake is available")
            
            return True
        except ImportError as e:
            self.log_message(f"✗ Missing dependency: {e}", "ERROR")
            return False

    def setup_llama_cpp(self) -> bool:
        """Setup llama.cpp if not already done"""
        try:
            if not self.llama_cpp_dir.exists():
                self.log_message("📦 Cloning llama.cpp repository...")
                success = self.run_subprocess_with_logging([
                    "git", "clone",
                    "https://github.com/ggerganov/llama.cpp",
                    str(self.llama_cpp_dir)
                ])
                if not success:
                    return False
            else:
                self.log_message("📦 llama.cpp directory exists, updating...")
                success = self.run_subprocess_with_logging([
                    "git", "pull"
                ], cwd=str(self.llama_cpp_dir))
                if not success:
                    self.log_message("⚠ Git pull failed, continuing with existing version", "WARNING")

            # Build llama.cpp
            self.log_message("🔨 Building llama.cpp...")
            build_dir = self.llama_cpp_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Configure with CMake
            success = self.run_subprocess_with_logging([
                "cmake", "..", "-DLLAMA_CURL=ON"
            ], cwd=str(build_dir))
            
            if not success:
                return False
            
            # Build
            success = self.run_subprocess_with_logging([
                "cmake", "--build", ".", "--config", "Release"
            ], cwd=str(build_dir))
            
            if success:
                self.log_message("✅ llama.cpp setup completed")
                return True
            else:
                return False
                
        except Exception as e:
            self.log_message(f"❌ Error setting up llama.cpp: {e}", "ERROR")
            return False

    def download_model(self, model_name: str) -> str:
        """Download a Hugging Face model"""
        model_dir = self.original_models_dir / model_name.replace("/", "_")
        
        if model_dir.exists():
            self.log_message(f"📁 Model {model_name} already exists locally")
            return str(model_dir)
        
        try:
            self.log_message(f"⬇️ Downloading model: {model_name}")
            self.current_status = f"Downloading {model_name}..."
            
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
            
            self.log_message(f"✅ Model downloaded successfully")
            return str(model_dir)
        except Exception as e:
            self.log_message(f"❌ Error downloading model: {e}", "ERROR")
            raise

    def convert_to_gguf(self, model_dir: str, model_name: str) -> str:
        """Convert model to GGUF format"""
        try:
            output_dir = self.quantized_models_dir / model_name.replace("/", "_")
            output_dir.mkdir(exist_ok=True)
            
            fp16_path = output_dir / "FP16.gguf"
            
            if fp16_path.exists():
                self.log_message("📄 F16 GGUF already exists, skipping conversion")
                return str(fp16_path)
            
            self.log_message("🔄 Converting model to GGUF format...")
            self.current_status = "Converting to GGUF..."
            
            # Use the convert script from llama.cpp
            convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
            
            success = self.run_subprocess_with_logging([
                sys.executable, str(convert_script),
                model_dir,
                "--outtype", "f16", 
                "--outfile", str(fp16_path)
            ])
            
            if success:
                self.log_message("✅ Model converted to GGUF format")
                return str(fp16_path)
            else:
                raise Exception("GGUF conversion failed")
                
        except Exception as e:
            self.log_message(f"❌ Error converting to GGUF: {e}", "ERROR")
            raise

    def quantize_model(self, fp16_path: str, methods: List[str], model_name: str) -> List[str]:
        """Quantize model using specified methods"""
        quantized_files = []
        output_dir = Path(fp16_path).parent
        
        # Get quantize binary path
        quantize_bin = self.llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            # Try alternative paths
            alt_paths = [
                self.llama_cpp_dir / "build" / "bin" / "quantize",
                self.llama_cpp_dir / "llama-quantize",
                self.llama_cpp_dir / "quantize"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    quantize_bin = alt_path
                    break
        
        if not quantize_bin.exists():
            self.log_message("❌ Quantize binary not found", "ERROR")
            return quantized_files
        
        for method in methods:
            if self.stop_event.is_set():
                break
                
            if method == "f16":
                # F16 is already created during conversion
                quantized_files.append(fp16_path)
                continue
            
            output_path = output_dir / f"{method.upper()}.gguf"
            
            if output_path.exists():
                self.log_message(f"📄 {method.upper()}.gguf already exists, skipping")
                quantized_files.append(str(output_path))
                continue
            
            try:
                self.log_message(f"⚙️ Quantizing with method: {method}")
                self.current_status = f"Quantizing with {method}..."
                
                success = self.run_subprocess_with_logging([
                    str(quantize_bin),
                    fp16_path,
                    str(output_path),
                    method
                ])
                
                if success and output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    self.log_message(f"✅ Created {method.upper()}.gguf ({size_mb:.1f} MB)")
                    quantized_files.append(str(output_path))
                else:
                    self.log_message(f"❌ Failed to create {method.upper()}.gguf", "ERROR")
                    
            except Exception as e:
                self.log_message(f"❌ Error quantizing with {method}: {e}", "ERROR")
        
        return quantized_files

    def stop_process(self):
        """Stop the current quantization process"""
        self.stop_event.set()
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
            except:
                try:
                    self.current_process.kill()
                except:
                    pass
        self.log_message("⏹️ Process stopped by user")

    def upload_to_huggingface(self, files: List[str], repo_id: str, token: str, private: bool = False):
        """Upload quantized files to Hugging Face"""
        try:
            self.log_message("🚀 Uploading to Hugging Face...")
            self.current_status = "Uploading to Hugging Face..."
            
            # Login with token
            login(token=token)
            
            # Create repository
            create_repo(repo_id, private=private, exist_ok=True)
            self.log_message(f"📁 Repository created/updated: {repo_id}")
            
            # Upload files
            api = HfApi()
            for file_path in files:
                if self.stop_event.is_set():
                    break
                    
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    self.log_message(f"📤 Uploading {filename}...")
                    
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type="model"
                    )
            
            if not self.stop_event.is_set():
                self.log_message("✅ All files uploaded successfully!")
                self.log_message(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            self.log_message(f"❌ Error uploading to Hugging Face: {e}", "ERROR")
            raise

    def convert_model(self, model_name: str, quant_methods: List[str],
                     upload_to_hf: bool = False, hf_token: str = "",
                     hf_repo_id: str = "", private_repo: bool = False,
                     base_model_dir: str = None, quantized_dir: str = None) -> Tuple[str, str, List[str]]:
        """Main function to convert a model"""
        if self.is_processing:
            return "❌ Another conversion is in progress", self.get_log_text(), []
        
        if not model_name.strip():
            return "❌ Please provide a model name", self.get_log_text(), []
        
        if not quant_methods:
            return "❌ Please select at least one quantization method", self.get_log_text(), []
        
        self.is_processing = True
        self.stop_event.clear()
        self.clear_log()
        
        # Update directories if provided
        if base_model_dir or quantized_dir:
            self.update_directories(base_model_dir, quantized_dir)
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return "❌ Missing dependencies", self.get_log_text(), []
            
            # Setup llama.cpp
            if not self.setup_llama_cpp():
                return "❌ Failed to setup llama.cpp", self.get_log_text(), []
            
            if self.stop_event.is_set():
                return "⏹️ Process stopped", self.get_log_text(), []
            
            # Download model
            self.current_status = "Downloading model..."
            model_dir = self.download_model(model_name)
            
            if self.stop_event.is_set():
                return "⏹️ Process stopped", self.get_log_text(), []
            
            # Convert to GGUF
            self.current_status = "Converting to GGUF..."
            fp16_path = self.convert_to_gguf(model_dir, model_name)
            
            if self.stop_event.is_set():
                return "⏹️ Process stopped", self.get_log_text(), []
            
            # Quantize
            self.current_status = "Quantizing model..."
            quantized_files = self.quantize_model(fp16_path, quant_methods, model_name)
            
            # Upload to Hugging Face if requested
            if upload_to_hf and hf_token and hf_repo_id and not self.stop_event.is_set():
                self.upload_to_huggingface(quantized_files, hf_repo_id, hf_token, private_repo)
            
            if self.stop_event.is_set():
                self.current_status = "⏹️ Process stopped"
            else:
                self.current_status = "✅ Conversion completed successfully!"
                self.log_message("🎉 All tasks completed successfully!")
            
            return self.current_status, self.get_log_text(), quantized_files
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.current_status = error_msg
            return error_msg, self.get_log_text(), []
        finally:
            self.is_processing = False
            self.stop_event.clear()

    def get_available_methods(self) -> List[Tuple[str, str]]:
        """Get list of available quantization methods"""
        return [(f"{method} - {desc}", method) for method, desc in self.quant_methods.items()]

    def get_status(self) -> str:
        """Get current status"""
        return self.current_status

    def is_busy(self) -> bool:
        """Check if currently processing"""
        return self.is_processing
