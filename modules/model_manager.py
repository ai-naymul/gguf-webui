#!/usr/bin/env python3
"""
Model Manager Module
Handles model management operations for the GGUF WebUI
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, base_dir: str = "./models"):
        # Ensure we always work with absolute paths to avoid confusion
        self.base_dir = Path(base_dir).resolve()
        self.quantized_models_dir = self.base_dir / "quantized" 
        self.uploaded_models_dir = self.base_dir / "uploaded"
        self.metadata_file = self.base_dir / "model_metadata.json"
        
        print(f"ModelManager initialized with:")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Quantized directory: {self.quantized_models_dir}")
        print(f"  Uploaded directory: {self.uploaded_models_dir}")
        
        # Create directories with proper error handling
        try:
            self.quantized_models_dir.mkdir(parents=True, exist_ok=True)
            self.uploaded_models_dir.mkdir(parents=True, exist_ok=True)
            print("Directories created successfully")
        except Exception as e:
            print(f"Error creating directories: {e}")
        
        # Load metadata
        self.metadata = self.load_metadata()
    
    def update_directories(self, base_dir: str):
        """Update directory paths and recreate manager"""
        self.__init__(base_dir)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def save_metadata(self):
        """Save model metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def scan_models(self) -> List[Dict[str, Any]]:
        """Enhanced model scanning with recursive search"""
        models = []
        
        print(f"Scanning quantized models in: {self.quantized_models_dir}")
        print(f"Scanning uploaded models in: {self.uploaded_models_dir}")
        
        # Scan quantized models with recursive search
        try:
            if self.quantized_models_dir.exists():
                print(f"Quantized directory exists: {self.quantized_models_dir}")
                
                # Search for .gguf files recursively
                gguf_files = list(self.quantized_models_dir.rglob("*.gguf"))
                print(f"Found {len(gguf_files)} .gguf files in quantized directory")
                
                for gguf_file in gguf_files:
                    model_info = self.get_model_info(str(gguf_file))
                    if model_info:
                        models.append(model_info)
                        print(f"Added model: {gguf_file.name}")
            else:
                print(f"Quantized directory does not exist: {self.quantized_models_dir}")
                self.quantized_models_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created quantized directory: {self.quantized_models_dir}")
        
        except Exception as e:
            print(f"Error scanning quantized models: {e}")
        
        # Scan uploaded models
        try:
            if self.uploaded_models_dir.exists():
                print(f"Uploaded directory exists: {self.uploaded_models_dir}")
                
                gguf_files = list(self.uploaded_models_dir.glob("*.gguf"))
                print(f"Found {len(gguf_files)} .gguf files in uploaded directory")
                
                for gguf_file in gguf_files:
                    model_info = self.get_model_info(str(gguf_file))
                    if model_info:
                        models.append(model_info)
                        print(f"Added uploaded model: {gguf_file.name}")
            else:
                print(f"Uploaded directory does not exist: {self.uploaded_models_dir}")
                self.uploaded_models_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created uploaded directory: {self.uploaded_models_dir}")
        
        except Exception as e:
            print(f"Error scanning uploaded models: {e}")
        
        print(f"Total models found: {len(models)}")
        return sorted(models, key=lambda x: x.get('name', ''))

    def debug_model_scanning(self):
        """Comprehensive debug method for model scanning issues"""
        debug_info = []
        
        debug_info.append("=== MODEL SCANNING DEBUG ===")
        debug_info.append(f"Current working directory: {Path.cwd()}")
        debug_info.append(f"Base directory (relative): {self.base_dir}")
        debug_info.append(f"Base directory (absolute): {self.base_dir.resolve()}")
        
        # Check each directory
        for dir_name, dir_path in [
            ("Quantized", self.quantized_models_dir),
            ("Uploaded", self.uploaded_models_dir)
        ]:
            debug_info.append(f"\n--- {dir_name} Directory Analysis ---")
            debug_info.append(f"Path: {dir_path}")
            debug_info.append(f"Absolute path: {dir_path.resolve()}")
            debug_info.append(f"Exists: {dir_path.exists()}")
            
            if dir_path.exists():
                debug_info.append(f"Is directory: {dir_path.is_dir()}")
                debug_info.append(f"Readable: {os.access(dir_path, os.R_OK)}")
                
                try:
                    all_items = list(dir_path.iterdir())
                    debug_info.append(f"Total items: {len(all_items)}")
                    
                    for item in all_items:
                        item_type = "📁 DIR" if item.is_dir() else "📄 FILE"
                        debug_info.append(f"  {item_type}: {item.name}")
                        
                        if item.is_file() and item.suffix.lower() == '.gguf':
                            debug_info.append(f"    ✅ GGUF FILE FOUND: {item.name}")
                    
                    # Test glob patterns
                    direct_gguf = list(dir_path.glob("*.gguf"))
                    recursive_gguf = list(dir_path.rglob("*.gguf"))
                    
                    debug_info.append(f"Direct *.gguf search: {len(direct_gguf)} files")
                    debug_info.append(f"Recursive **/*.gguf search: {len(recursive_gguf)} files")
                    
                    for gguf_file in recursive_gguf:
                        debug_info.append(f"  Found: {gguf_file.relative_to(dir_path)}")
                    
                except Exception as e:
                    debug_info.append(f"  ❌ Error reading directory: {e}")
        
        return "\n".join(debug_info)


    
    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        try:
            path = Path(model_path)
            if not path.exists():
                return None
            
            # Get file stats
            stat = path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created_time = time.ctime(stat.st_ctime)
            
            # Get model type from filename
            model_type = "Unknown"
            filename = path.stem.upper()
            
            if "Q2_K" in filename:
                model_type = "Q2_K (2-bit)"
            elif "Q3_K" in filename:
                model_type = "Q3_K (3-bit)"
            elif "Q4_K" in filename:
                model_type = "Q4_K (4-bit)"
            elif "Q5_K" in filename:
                model_type = "Q5_K (5-bit)"
            elif "Q6_K" in filename:
                model_type = "Q6_K (6-bit)"
            elif "Q8_0" in filename:
                model_type = "Q8_0 (8-bit)"
            elif "F16" in filename:
                model_type = "F16 (16-bit)"
            elif "IQ4_XS" in filename:
                model_type = "IQ4_XS (4-bit)"
            
            # Get metadata if available
            metadata = self.metadata.get(str(path), {})
            
            return {
                "name": path.name,
                "path": str(path),
                "size_mb": size_mb,
                "size_str": f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB",
                "type": model_type,
                "created": created_time,
                "parent_model": metadata.get("parent_model", "Unknown"),
                "quantization_date": metadata.get("quantization_date", created_time),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", [])
            }
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_path}: {e}")
            return None
    
    def get_available_models(self) -> Tuple[List[List[str]], List[str]]:
        """Get available models for dropdown and table display"""
        models = self.scan_models()
        
        # For table display
        table_data = []
        for model in models:
            table_data.append([
                model["name"],
                model["size_str"],
                model["type"],
                model["created"]
            ])
        
        # For dropdown
        dropdown_choices = [model["path"] for model in models]
        
        return table_data, dropdown_choices
    
    def get_model_list(self) -> Tuple[List[List[str]], List[str]]:
        """Get model list for management interface"""
        return self.get_available_models()
    
    def delete_model(self, model_path: str) -> Tuple[List[List[str]], List[str]]:
        """Delete a model file"""
        try:
            if not model_path:
                return self.get_available_models()
            
            path = Path(model_path)
            if path.exists():
                path.unlink()
                
                # Remove from metadata
                if model_path in self.metadata:
                    del self.metadata[model_path]
                    self.save_metadata()
                
                logger.info(f"Deleted model: {path.name}")
            
            return self.get_available_models()
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return self.get_available_models()
    
    def get_model_detailed_info(self, model_path: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            if not model_path:
                return {}
            
            model_info = self.get_model_info(model_path)
            if not model_info:
                return {"error": "Model not found"}
            
            # Add additional details
            path = Path(model_path)
            
            # Try to get more technical details
            detailed_info = {
                "basic_info": {
                    "name": model_info["name"],
                    "path": model_info["path"],
                    "size": model_info["size_str"],
                    "type": model_info["type"],
                    "created": model_info["created"]
                },
                "technical_info": {
                    "file_size_bytes": path.stat().st_size,
                    "format": "GGUF",
                    "quantization": model_info["type"],
                    "parent_model": model_info["parent_model"]
                },
                "metadata": {
                    "description": model_info["description"],
                    "tags": model_info["tags"],
                    "quantization_date": model_info["quantization_date"]
                }
            }
            
            return detailed_info
            
        except Exception as e:
            logger.error(f"Error getting detailed model info: {e}")
            return {"error": str(e)}
    
    def add_model_metadata(self, model_path: str, metadata: Dict[str, Any]):
        """Add metadata for a model"""
        try:
            self.metadata[model_path] = metadata
            self.save_metadata()
            logger.info(f"Added metadata for {model_path}")
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
    
    def update_model_metadata(self, model_path: str, **kwargs):
        """Update metadata for a model"""
        try:
            if model_path not in self.metadata:
                self.metadata[model_path] = {}
            
            self.metadata[model_path].update(kwargs)
            self.save_metadata()
            logger.info(f"Updated metadata for {model_path}")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def get_storage_info(self) -> str:
        """Get storage information"""
        try:
            models = self.scan_models()
            total_size = sum(model["size_mb"] for model in models)
            
            # Get directory sizes
            quantized_size = self.get_directory_size(self.quantized_models_dir)
            uploaded_size = self.get_directory_size(self.uploaded_models_dir)
            
            storage_info = f"""
📊 Storage Information:

📁 Total Models: {len(models)}
💾 Total Size: {total_size/1024:.2f} GB ({total_size:.1f} MB)

📂 Directory Breakdown:
  • Quantized Models: {quantized_size/1024:.2f} GB
  • Uploaded Models: {uploaded_size/1024:.2f} GB

🗂️ Model Types:
"""
            
            # Count by type
            type_counts = {}
            for model in models:
                model_type = model["type"]
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            for model_type, count in sorted(type_counts.items()):
                storage_info += f"  • {model_type}: {count} models\n"
            
            return storage_info.strip()
            
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return f"Error getting storage info: {e}"
    
    def get_directory_size(self, directory: Path) -> float:
        """Get total size of a directory in MB"""
        try:
            total_size = 0
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
            return 0.0
    
    def cleanup_old_models(self, days_old: int = 30) -> str:
        """Clean up old models"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            models = self.scan_models()
            deleted_count = 0
            deleted_size = 0
            
            for model in models:
                path = Path(model["path"])
                if path.stat().st_mtime < cutoff_time:
                    deleted_size += model["size_mb"]
                    path.unlink()
                    deleted_count += 1
                    
                    # Remove from metadata
                    if model["path"] in self.metadata:
                        del self.metadata[model["path"]]
            
            if deleted_count > 0:
                self.save_metadata()
            
            return f"🗑️ Cleaned up {deleted_count} models ({deleted_size:.1f} MB)"
            
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
            return f"Error during cleanup: {e}"
    
    def export_model_list(self) -> str:
        """Export model list to CSV"""
        try:
            models = self.scan_models()
            
            # Create export directory
            export_dir = self.base_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"model_list_{timestamp}.csv"
            filepath = export_dir / filename
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("Name,Path,Size(MB),Type,Created,Parent Model\n")
                for model in models:
                    f.write(f'"{model["name"]}","{model["path"]}",{model["size_mb"]:.1f},"{model["type"]}","{model["created"]}","{model["parent_model"]}"\n')
            
            return f"✅ Model list exported to: {filepath}"
            
        except Exception as e:
            logger.error(f"Error exporting model list: {e}")
            return f"Error exporting: {e}"
    
    def import_model(self, file_path: str, model_name: str = "", description: str = "") -> str:
        """Import a model file"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return "❌ Source file not found"
            
            # Determine destination
            if model_name:
                dest_name = f"{model_name}.gguf"
            else:
                dest_name = source_path.name
            
            dest_path = self.uploaded_models_dir / dest_name
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Add metadata
            self.add_model_metadata(str(dest_path), {
                "description": description,
                "import_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_path": str(source_path)
            })
            
            return f"✅ Model imported successfully: {dest_name}"
            
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            return f"❌ Import error: {e}"