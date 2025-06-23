#!/bin/bash

# GGUF Model Quantization Script
# This script quantizes Hugging Face models to various GGUF formats

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_NAME=""
HF_REPO_ID=""
BASE_MODEL_DIR="./original_model"
QUANTIZED_DIR="./quantized_model"
LLAMA_CPP_DIR="./llama.cpp"
UPLOAD_TO_HF=false
PRIVATE_REPO=false

# Available quantization methods with descriptions
declare -A QUANT_METHODS=(
    ["q2_k"]="2-bit quantization (1.49 GB)"
    ["q3_k_s"]="3-bit quantization - Small (1.71 GB)"
    ["q3_k_m"]="3-bit quantization - Medium (1.86 GB)" 
    ["q3_k_l"]="3-bit quantization - Large (1.98 GB)"
    ["iq4_xs"]="4-bit quantization - Extra Small (2.05 GB)"
    ["q4_k_s"]="4-bit quantization - Small (2.15 GB)"
    ["q4_k_m"]="4-bit quantization - Medium (2.24 GB)"
    ["q5_k_s"]="5-bit quantization - Small (2.54 GB)"
    ["q5_k_m"]="5-bit quantization - Medium (2.59 GB)"
    ["q6_k"]="6-bit quantization (2.97 GB)"
    ["q8_0"]="8-bit quantization (3.84 GB)"
    ["f16"]="16-bit float (7.22 GB)"
)

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

GGUF Model Quantization Script

Required Options:
    -m, --model MODEL_NAME          Hugging Face model name (e.g., "Qwen/Qwen1.5-1.8B")

Optional Options:
    -q, --quant METHODS             Comma-separated quantization methods (default: q4_k_m)
    -o, --output DIR               Output directory for quantized models (default: ./quantized_model)
    -b, --base DIR                 Base model directory (default: ./original_model)
    -l, --llama-cpp DIR            llama.cpp directory (default: ./llama.cpp)
    -u, --upload                   Upload to Hugging Face Hub
    -r, --repo REPO_ID             Hugging Face repository ID for upload
    -p, --private                  Make repository private (default: public)
    --list-methods                 List available quantization methods
    --all                          Use all available quantization methods
    -h, --help                     Show this help message

Available Quantization Methods:
EOF
    for method in $(printf '%s\n' "${!QUANT_METHODS[@]}" | sort); do
        printf "    %-10s %s\n" "$method" "${QUANT_METHODS[$method]}"
    done
    
    cat << EOF

Examples:
    # Basic quantization with default q4_k_m
    $0 -m "Qwen/Qwen1.5-1.8B"
    
    # Multiple quantization methods
    $0 -m "Qwen/Qwen1.5-1.8B" -q "q4_k_m,q8_0,f16"
    
    # All quantization methods
    $0 -m "Qwen/Qwen1.5-1.8B" --all
    
    # With Hugging Face upload
    $0 -m "Qwen/Qwen1.5-1.8B" -q "q4_k_m,q8_0" -u -r "username/model-name"

EOF
}

# Function to list available methods
list_methods() {
    print_color $BLUE "Available Quantization Methods:"
    echo
    for method in $(printf '%s\n' "${!QUANT_METHODS[@]}" | sort); do
        printf "%-10s %s\n" "$method" "${QUANT_METHODS[$method]}"
    done
}

# Function to validate quantization method
validate_method() {
    local method=$1
    if [[ -z "${QUANT_METHODS[$method]}" ]]; then
        print_color $RED "Error: Invalid quantization method '$method'"
        print_color $YELLOW "Run '$0 --list-methods' to see available methods"
        exit 1
    fi
}

# Function to setup llama.cpp
setup_llama_cpp() {
    print_color $BLUE "Setting up llama.cpp..."
    
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        print_color $YELLOW "Cloning llama.cpp repository..."
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    else
        print_color $YELLOW "llama.cpp directory already exists, updating..."
        cd "$LLAMA_CPP_DIR"
        git pull
        cd ..
    fi
    
    cd "$LLAMA_CPP_DIR"
    print_color $YELLOW "Building llama.cpp..."
    LLAMA_CUBLAS=1 make
    
    if [ -f "requirements.txt" ]; then
        print_color $YELLOW "Installing Python requirements..."
        pip install -r requirements.txt
    fi
    cd ..
}

# Function to check dependencies
check_dependencies() {
    print_color $BLUE "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_color $RED "Error: Python is not installed"
        exit 1
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        print_color $RED "Error: Git is not installed"
        exit 1
    fi
    
    # Check if huggingface_hub is installed
    if ! python3 -c "import huggingface_hub" 2>/dev/null && ! python -c "import huggingface_hub" 2>/dev/null; then
        print_color $YELLOW "Installing huggingface_hub..."
        pip install huggingface_hub
    fi
}

# Function to download model
download_model() {
    print_color $BLUE "Downloading model: $MODEL_NAME"
    
    mkdir -p "$BASE_MODEL_DIR"
    
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_NAME', local_dir='$BASE_MODEL_DIR', local_dir_use_symlinks=False)
print('Model downloaded successfully!')
" || python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_NAME', local_dir='$BASE_MODEL_DIR', local_dir_use_symlinks=False)
print('Model downloaded successfully!')
"
}

# Function to convert to GGUF
convert_to_gguf() {
    print_color $BLUE "Converting model to GGUF format..."
    
    mkdir -p "$QUANTIZED_DIR"
    
    local fp16_path="$QUANTIZED_DIR/FP16.gguf"
    
    if [ ! -f "$fp16_path" ]; then
        print_color $YELLOW "Converting to F16 GGUF..."
        python3 "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" "$BASE_MODEL_DIR" --outtype f16 --outfile "$fp16_path" || \
        python "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" "$BASE_MODEL_DIR" --outtype f16 --outfile "$fp16_path"
    else
        print_color $YELLOW "F16 GGUF already exists, skipping conversion..."
    fi
}

# Function to quantize model
quantize_model() {
    local methods=("$@")
    local fp16_path="$QUANTIZED_DIR/FP16.gguf"
    
    print_color $BLUE "Starting quantization process..."
    
    for method in "${methods[@]}"; do
        print_color $YELLOW "Quantizing with method: $method (${QUANT_METHODS[$method]})"
        
        local output_path="$QUANTIZED_DIR/${method^^}.gguf"
        
        if [ -f "$output_path" ]; then
            print_color $YELLOW "File $output_path already exists, skipping..."
            continue
        fi
        
        if [ "$method" = "f16" ]; then
            # F16 is already created during conversion
            print_color $GREEN "F16 quantization completed (original FP16 format)"
            continue
        fi
        
        print_color $YELLOW "Running quantization: $method"
        "$LLAMA_CPP_DIR/quantize" "$fp16_path" "$output_path" "$method"
        
        if [ $? -eq 0 ]; then
            print_color $GREEN "✓ Successfully created: $output_path"
            # Show file size
            if command -v du &> /dev/null; then
                local size=$(du -h "$output_path" | cut -f1)
                print_color $GREEN "  File size: $size"
            fi
        else
            print_color $RED "✗ Failed to create: $output_path"
        fi
    done
}

# Function to test model
test_model() {
    local method=$1
    local model_path="$QUANTIZED_DIR/${method^^}.gguf"
    
    if [ -f "$model_path" ]; then
        print_color $BLUE "Testing model: $model_path"
        
        # Check if chat prompt exists
        local prompt_file="$LLAMA_CPP_DIR/prompts/chat-with-bob.txt"
        if [ -f "$prompt_file" ]; then
            "$LLAMA_CPP_DIR/main" -m "$model_path" -n 50 --repeat_penalty 1.0 --color -i -r "User:" -f "$prompt_file" || true
        else
            "$LLAMA_CPP_DIR/main" -m "$model_path" -n 50 --repeat_penalty 1.0 --color -p "Hello, how are you?" || true
        fi
    fi
}

# Function to upload to Hugging Face
upload_to_hf() {
    if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ]; then
        print_color $BLUE "Uploading to Hugging Face Hub..."
        
        python3 -c "
from huggingface_hub import HfApi, create_repo
import os
import glob

try:
    # Create repository
    repo_url = create_repo('$HF_REPO_ID', private=$PRIVATE_REPO, exist_ok=True)
    print(f'Repository created/updated: {repo_url}')
    
    # Upload files
    api = HfApi()
    quantized_files = glob.glob('$QUANTIZED_DIR/*.gguf')
    
    for file_path in quantized_files:
        filename = os.path.basename(file_path)
        print(f'Uploading {filename}...')
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id='$HF_REPO_ID',
            repo_type='model',
        )
        print(f'✓ Uploaded {filename}')
    
    print('All files uploaded successfully!')
    
except Exception as e:
    print(f'Error uploading to Hugging Face: {e}')
    exit(1)
" || python -c "
from huggingface_hub import HfApi, create_repo
import os
import glob

try:
    # Create repository
    repo_url = create_repo('$HF_REPO_ID', private=$PRIVATE_REPO, exist_ok=True)
    print(f'Repository created/updated: {repo_url}')
    
    # Upload files
    api = HfApi()
    quantized_files = glob.glob('$QUANTIZED_DIR/*.gguf')
    
    for file_path in quantized_files:
        filename = os.path.basename(file_path)
        print(f'Uploading {filename}...')
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id='$HF_REPO_ID',
            repo_type='model',
        )
        print(f'✓ Uploaded {filename}')
    
    print('All files uploaded successfully!')
    
except Exception as e:
    print(f'Error uploading to Hugging Face: {e}')
    exit(1)
"
    fi
}

# Function to show summary
show_summary() {
    print_color $GREEN "=== Quantization Summary ==="
    print_color $BLUE "Model: $MODEL_NAME"
    print_color $BLUE "Output directory: $QUANTIZED_DIR"
    
    if [ -d "$QUANTIZED_DIR" ]; then
        print_color $BLUE "Generated files:"
        for file in "$QUANTIZED_DIR"/*.gguf; do
            if [ -f "$file" ]; then
                local filename=$(basename "$file")
                local size="Unknown"
                if command -v du &> /dev/null; then
                    size=$(du -h "$file" | cut -f1)
                fi
                printf "  %-15s %s\n" "$filename" "$size"
            fi
        done
    fi
    
    if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ]; then
        print_color $BLUE "Uploaded to: https://huggingface.co/$HF_REPO_ID"
    fi
}

# Parse command line arguments
METHODS=("q4_k_m")  # Default method
USE_ALL_METHODS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -q|--quant)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --all)
            USE_ALL_METHODS=true
            shift
            ;;
        -o|--output)
            QUANTIZED_DIR="$2"
            shift 2
            ;;
        -b|--base)
            BASE_MODEL_DIR="$2"
            shift 2
            ;;
        -l|--llama-cpp)
            LLAMA_CPP_DIR="$2"
            shift 2
            ;;
        -u|--upload)
            UPLOAD_TO_HF=true
            shift
            ;;
        -r|--repo)
            HF_REPO_ID="$2"
            shift 2
            ;;
        -p|--private)
            PRIVATE_REPO=true
            shift
            ;;
        --list-methods)
            list_methods
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    print_color $RED "Error: Model name is required"
    show_usage
    exit 1
fi

# Set all methods if requested
if [ "$USE_ALL_METHODS" = true ]; then
    METHODS=($(printf '%s\n' "${!QUANT_METHODS[@]}" | sort))
fi

# Validate quantization methods
for method in "${METHODS[@]}"; do
    validate_method "$method"
done

# Validate upload requirements
if [ "$UPLOAD_TO_HF" = true ] && [ -z "$HF_REPO_ID" ]; then
    print_color $RED "Error: Repository ID is required for upload (-r/--repo)"
    exit 1
fi

# Main execution
print_color $GREEN "=== GGUF Model Quantization Script ==="
print_color $BLUE "Model: $MODEL_NAME"
print_color $BLUE "Methods: ${METHODS[*]}"
print_color $BLUE "Output: $QUANTIZED_DIR"

# Run the process
check_dependencies
setup_llama_cpp
download_model
convert_to_gguf
quantize_model "${METHODS[@]}"

# Optional: Test the first quantized model
if [ ${#METHODS[@]} -gt 0 ]; then
    print_color $BLUE "Testing first quantized model..."
    test_model "${METHODS[0]}"
fi

# Upload if requested
upload_to_hf

# Show summary
show_summary

print_color $GREEN "=== Quantization completed successfully! ==="