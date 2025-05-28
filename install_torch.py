import platform
import subprocess
import sys

def check_and_install_pytorch():
    system = platform.system()
    
    # Check for NVIDIA GPU (primarily for Linux/Windows)
    has_nvidia_gpu = False
    if system == "Linux" or system == "Windows":
        try:
            # Try running nvidia-smi. If it runs without error, assume CUDA is available.
            # Redirect stdout and stderr to prevent output unless there's an error we need to catch
            subprocess.check_output("nvidia-smi", stderr=subprocess.STDOUT) 
            has_nvidia_gpu = True
            print("NVIDIA GPU detected.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NVIDIA GPU not detected or nvidia-smi not found.")
            
    # Check for Apple Silicon (macOS)
    is_apple_silicon = False
    if system == "Darwin": # Darwin is the system name for macOS
        processor = platform.processor()
        if "arm" in processor.lower():
            is_apple_silicon = True
            print("Apple Silicon (MPS) detected.")
        else:
            print("Intel Mac detected.")

    # Determine the correct install command
    if has_nvidia_gpu:
        print("Installing PyTorch for CUDA (cu126)...")
        # --- Updated command as requested ---
        command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu126"]
        print("Installing xformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xformers"])
    elif is_apple_silicon:
        print("Installing standard PyTorch with MPS support...")
        command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else: # Default to standard CPU version for Intel Macs or other systems
        print("Installing standard PyTorch (CPU)...")
        command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        
    # Execute the command
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.check_call(command)
        print("PyTorch installation successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1) # Exit if installation fails

if __name__ == "__main__":
    check_and_install_pytorch()
