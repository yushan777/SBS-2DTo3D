import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import time
from safetensors.torch import load_file as load_safetensors # Import safetensors loading function
import matplotlib as mpl # Import matplotlib for colormap
# import matplotlib.pyplot as plt  # Import matplotlib for colormap (old)
# from matplotlib import cm  # Import colormap module

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

def load_model(model_name, device, models_dir='models/depthanything'):
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

def process_image(model, image_tensor, device, dtype, is_metric, output_filename_base):
    """Processes a single image to estimate depth."""
    # Preprocessing similar to the original node
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # image_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

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

    # COLOR DEPTH MAP
    # Apply colormap (Spectral_r to match original implementation)
    cmap = mpl.colormaps['Spectral_r']
    colored_depth = cmap(depth_np)[:, :, :3]  # Remove alpha channel
    colored_depth = (colored_depth * 255).astype(np.uint8)
    # Convert to PIL image
    colored_depth_image = Image.fromarray(colored_depth)

    # save the image(s) into the output directory before returning
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save grayscale depth map
    grayscale_path = os.path.join(output_dir, f"{output_filename_base}_depth.png")
    depth_image.save(grayscale_path)
    print(f"Saved grayscale depth map to: {grayscale_path}")
    
    # Save colored depth map
    colored_path = os.path.join(output_dir, f"{output_filename_base}_depth_colored.png")
    colored_depth_image.save(colored_path)
    print(f"Saved colored depth map to: {colored_path}")
    
    return depth_image, colored_depth_image

def generate_depth_maps(input_image, model_name):
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
        model, dtype, is_metric = load_model(model_name, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(input_image).unsqueeze(0).to(device=device, dtype=dtype)

    # Generate output filename base (use timestamp since Gradio doesn't provide original filename)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = f"gradio_image_{timestamp}"
    
    # Process image
    try:
        depth_image, colored_depth_image = process_image(model, image_tensor, device, dtype, is_metric, output_filename_base)
        return depth_image, colored_depth_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

iface = gr.Interface(
    fn=generate_depth_maps,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(choices=AVAILABLE_MODELS, label="Select Model", value='depth_anything_v2_vitl_fp16.safetensors')
    ],
    outputs=[
        gr.Image(type="pil", label="Grayscale Depth Map"),
        gr.Image(type="pil", label="Color Depth Map")
    ],
    title="Depth Anything V2 Gradio Demo"
)

if __name__ == "__main__":
    iface.launch()
