import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import argparse
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import time
from safetensors.torch import load_file as load_safetensors # Import safetensors loading function
import matplotlib as mpl # Import matplotlib for colormap

# Assuming the depth_anything_v2 directory is in the same folder as main.py
from depth_anything_v2.dpt import DepthAnythingV2

# Model configurations (similar to the original node)
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Available models and their Hugging Face filenames
AVAILABLE_MODELS = [
    'depth_anything_v2_vits_fp16.safetensors',
    'depth_anything_v2_vits_fp32.safetensors',
    'depth_anything_v2_vitb_fp16.safetensors',
    'depth_anything_v2_vitb_fp32.safetensors',
    'depth_anything_v2_vitl_fp16.safetensors',
    'depth_anything_v2_vitl_fp32.safetensors',
    'depth_anything_v2_metric_hypersim_vitl_fp32.safetensors',
    'depth_anything_v2_metric_vkitti_vitl_fp32.safetensors'
]

def load_model(model_name, device, models_dir='models'):
    """Loads the specified Depth Anything V2 model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

    print(f"Selected model: {model_name}")
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    encoder = 'vitl' # Default
    if "vitl" in model_name:
        encoder = "vitl"
    elif "vitb" in model_name:
        encoder = "vitb"
    elif "vits" in model_name:
        encoder = "vits"

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} to {model_path}...")
        os.makedirs(models_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="yushan777/DepthAnythingV2",
                filename=model_name,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    print(f"Loading model from: {model_path}")
    # Use safetensors.torch.load_file for .safetensors files
    state_dict = load_safetensors(model_path, device='cpu')

    max_depth = 20.0 if "hypersim" in model_name else 80.0
    is_metric = 'metric' in model_name

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**{**config, 'is_metric': is_metric, 'max_depth': max_depth})

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device, dtype=dtype)
    print("Model loaded successfully.")

    return model, dtype, is_metric

def process_image(model, image_path, output_dir, device, dtype, is_metric):
    """Processes a single image to estimate depth."""
    print(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # Preprocessing similar to the original node
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

    # Ensure dimensions are divisible by 14
    orig_H, orig_W = image_tensor.shape[2:]
    new_H, new_W = orig_H, orig_W
    if new_W % 14 != 0:
        new_W = new_W - (new_W % 14)
    if new_H % 14 != 0:
        new_H = new_H - (new_H % 14)

    if new_H != orig_H or new_W != orig_W:
        print(f"Resizing input from {orig_W}x{orig_H} to {new_W}x{new_H}")
        image_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)

    # Inference
    start_time = time.time()
    if device.type == 'cuda': # Reset peak memory stats before inference if using CUDA
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        depth = model(image_tensor)
        
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds")

    if device.type == 'cuda': # Report peak CUDA memory usage after inference
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mib = peak_memory_bytes / (1024 * 1024) 
        peak_memory_gib = peak_memory_bytes / (1024 * 1024 * 1024)
        print(f"Peak GPU Memory Allocated during inference: {peak_memory_mib:.2f} MiB ({peak_memory_gib:.2f} GiB)")

    # Postprocessing
    depth = depth.squeeze(0).squeeze(0) # Remove batch (dim 0) and channel (dim 0 again after first squeeze) -> (H, W)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) # Normalize to 0-1 -> (H, W)

    # Resize back to original (or slightly adjusted) size
    # Ensure final dimensions are even for potential later use
    final_H = (orig_H // 2) * 2
    final_W = (orig_W // 2) * 2
    # Check shape using correct indices for 2D tensor
    if depth.ndim == 2 and (depth.shape[0] != final_H or depth.shape[1] != final_W):
         # Interpolate: expects NCHW, add N and C dims. Squeeze back to HW.
         depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(final_H, final_W), mode="bilinear", align_corners=False).squeeze()

    depth = torch.clamp(depth, 0, 1)

    if is_metric:
        depth = 1.0 - depth # Invert for metric models

    # Convert to numpy array and scale to 0-255
    depth_np = depth.cpu().numpy()
    depth_visual = (depth_np * 255).astype(np.uint8)

    # Create PIL image (grayscale)
    depth_image = Image.fromarray(depth_visual)

    # Save the output image
    try:
        # get just the input filename but with ext separate
        name, ext = os.path.splitext(os.path.basename(image_path))
        
        # GRAYSCALE DEPTH MAP
        # build the full grayscale_output_path
        grayscale_output_path = f'{output_dir}/{name}_depth_gray{ext}'        
        depth_image.save(grayscale_output_path)
        print(f"Depth map saved to: {grayscale_output_path}")

        # COLOR DEPTH MAP
        # Apply colormap (Spectral_r to match original implementation)
        cmap = mpl.get_cmap('Spectral_r')
        # cmap = cm.get_cmap('Spectral_r')
        colored_depth = cmap(depth_np)[:, :, :3]  # Remove alpha channel
        colored_depth = (colored_depth * 255).astype(np.uint8)
        # Convert to PIL image
        colored_depth_image = Image.fromarray(colored_depth)
        # Save colored depth map
        colored_output_path = f'{output_dir}/{name}_depth_color{ext}'
        colored_depth_image.save(colored_output_path)
        print(f"Colored depth map saved to: {colored_output_path}")

    except Exception as e:
        print(f"Error saving output image {grayscale_output_path} and {colored_output_path}: {e}")


# ================================================================================
def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 CLI Tool")
    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output-dir', type=str, default='output', help='Dir to save the output depth map.')
    parser.add_argument('--model', type=str, default='depth_anything_v2_vitl_fp16.safetensors',
                        choices=AVAILABLE_MODELS, help='Name of the model to use.')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to store/load downloaded models.')
    # --gpu flag removed, GPU is now default if available

    args = parser.parse_args()

    # Determine device (Default to GPU if available, fallback to CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected. Using GPU.")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Apple Silicon GPU detected. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected or available. Using CPU.")


    # Load model
    try:
        model, dtype, is_metric = load_model(args.model, device, args.models_dir)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Process image
    process_image(model, args.input, args.output_dir, device, dtype, is_metric)

if __name__ == "__main__":
    main()

"""
Usage: 
python3 main.py \
    --input path/to/image.png \
    --output-dir /path/to/saved/depthmaps \
    --model depth_anything_v2_vitl_fp16.safetensors \
    --models-dir /path/to/your/downloaded/models 

Example:

python3 main.py \
    --input 'input/bottle.png' \
    --output-dir 'output' \
    --model 'depth_anything_v2_vitl_fp32.safetensors' \
    --models-dir 'models' 

Note: Depth Anything expects input image dims to be divisible by 14.
If not then it will be resized down to nearest divisible dim.  
Final depthmap image will match original size
"""
