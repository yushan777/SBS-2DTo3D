import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import time
from utils.colored_print import color, style
from safetensors.torch import load_file as load_safetensors # Import safetensors loading function
import matplotlib as mpl # Import matplotlib for colormap
# import matplotlib.pyplot as plt  # Import matplotlib for colormap (old)
# from matplotlib import cm  # Import colormap module

# Assuming the depth_anything_v2 directory is in the same folder as main.py
from depth_anything_v2.dpt import DepthAnythingV2
from sbs.sbs import process_image_sbs # Import for SBS processing

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

# load depth-anything-2 model
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

def process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base):
    # performs the core inference, and post-processes the raw depth output by (normalization, resizing), 
    # converts it to a PIL image, and saves it.


    # Ensure dimensions are divisible by 14
    # if not, then they will be resize to the nearest multiple of 14.  
    # the final depth-map image will be resized to match dims of the original input image
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

    # # COLOR DEPTH MAP
    # # Apply colormap (Spectral_r to match original implementation)
    # cmap = mpl.colormaps['Spectral_r']
    # colored_depth = cmap(depth_np)[:, :, :3]  # Remove alpha channel
    # colored_depth = (colored_depth * 255).astype(np.uint8)
    # # Convert to PIL image
    # colored_depth_image = Image.fromarray(colored_depth)

    # save the image(s) into the output directory before returning
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save grayscale depth map
    grayscale_path = os.path.join(output_dir, f"{output_filename_base}_depth.png")
    depth_image.save(grayscale_path)
    print(f"Saved grayscale depth map to: {grayscale_path}")
    
    # Save colored depth map
    colored_path = os.path.join(output_dir, f"{output_filename_base}_depth_colored.png")
    # colored_depth_image.save(colored_path)
    # print(f"Saved colored depth map to: {colored_path}")
    
    return depth_image

def generate_depth_map_only(input_image, model_name):
    """Generates only the depth map from the input image."""
    if input_image is None:
        gr.Warning("Please upload an image for depth map generation.")
        return None
    if model_name is None:
        gr.Warning("Please select a model for depth map generation.")
        return None

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected for depth map. Using GPU.")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Apple Silicon GPU detected for depth map. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected for depth map. Using CPU.")

    # Load model
    try:
        model, dtype, is_metric = load_model(model_name, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        gr.Error(f"Failed to load model: {e}")
        return None

    # Preprocessing
    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image_tensor = transform_normalize(input_image).unsqueeze(0).to(device=device, dtype=dtype)
    except Exception as e:
        print(f"Error during image transformation: {e}")
        gr.Error(f"Error during image transformation: {e}")
        return None

    # Generate output filename base
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = f"gradio_image_{timestamp}"
    
    # Process image for depth map
    try:
        depth_image_pil = process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base)
        if depth_image_pil:
            print("Depth map generated successfully.")
        return depth_image_pil
    except Exception as e:
        print(f"Error processing image for depth map: {e}")
        gr.Error(f"Error processing image for depth map: {e}")
        return None

def generate_sbs_image_from_depth(original_input_image, depth_map_pil, model_name, sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength):
    """Generates the SBS 3D image using the original image and a pre-generated depth map."""
    if original_input_image is None:
        gr.Warning("Please provide the original input image for SBS generation.")
        return None
    if depth_map_pil is None:
        gr.Warning("Please generate or provide a depth map for SBS generation.")
        return None
    if model_name is None: # Needed for dtype
        gr.Warning("Please select a model (needed for data type).")
        return None


    # Determine device (can be different from depth map generation if run separately)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected for SBS. Using GPU.")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Apple Silicon GPU detected for SBS. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected for SBS. Using CPU.")

    # Determine dtype from model_name (as load_model is not called here directly for the primary model)
    dtype = torch.float16 if "fp16" in model_name else torch.float32

    try:
        # Prepare base_image for SBS: PIL to Tensor [1, H, W, C], float16, range [0,1]
        if original_input_image.mode != 'RGB':
            original_input_image = original_input_image.convert('RGB')
        
        transform_to_tensor = transforms.ToTensor() # Converts PIL [0,255] to Tensor [0,1]
        base_image_for_sbs = transform_to_tensor(original_input_image).permute(1, 2, 0).unsqueeze(0)
        # Ensure correct dtype for SBS processing, which expects float16
        base_image_for_sbs = base_image_for_sbs.to(device=device, dtype=torch.float16)


        # Prepare depth_map for SBS: PIL to Tensor [1, H, W, 1], float16, range [0,1]
        # depth_map_pil is already grayscale
        depth_map_for_sbs = transform_to_tensor(depth_map_pil).permute(1, 2, 0).unsqueeze(0)
        # Ensure correct dtype for SBS processing
        depth_map_for_sbs = depth_map_for_sbs.to(device=device, dtype=torch.float16)
        
        # Ensure depth_blur_strength is odd for SBS
        if sbs_depth_blur_strength % 2 == 0:
            sbs_depth_blur_strength +=1
            gr.Info(f"SBS Depth Blur Strength adjusted to {sbs_depth_blur_strength} (must be odd).")

        print(f"Calling process_image_sbs with method: {sbs_method}, scale: {sbs_depth_scale}, mode: {sbs_mode}, blur: {sbs_depth_blur_strength}")
        sbs_image_tensor = process_image_sbs(
                base_image=base_image_for_sbs,
                depth_map=depth_map_for_sbs,
                method=sbs_method,
                depth_scale=sbs_depth_scale,
                mode=sbs_mode,
                depth_blur_strength=sbs_depth_blur_strength
            )
        
        print(f"[run_gradio.generate_sbs_from_depth] sbs_image_tensor shape: {sbs_image_tensor.shape}", color.YELLOW)
        sbs_image_pil = transforms.ToPILImage()(sbs_image_tensor.squeeze(0).cpu().permute(2, 0, 1))
        print("SBS image generated successfully.")
        return sbs_image_pil
    
    except Exception as e:
        print(f"Error generating SBS image: {e}")
        gr.Error(f"Error generating SBS image: {e}")
        return None

# ================================================
# GRADIO UI
with gr.Blocks(title="SBS 2D To 3D") as demo:
    
    gr.Markdown("## SBS 2D To 3D Demo")

    with gr.Tabs():
        with gr.Tab("Image"):
            
            gr.Markdown("Upload an image, generate its depth map, then generate the 3D SBS image.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_component = gr.Image(label="Input Image", type="pil", height=480)
                    
                with gr.Column(scale=1):
                    output_grayscale_component = gr.Image(label="Generated Depth Map", type="pil", height=480, interactive=False) # Depth map is output here
                    
                with gr.Column(scale=1):
                    model_dropdown_component = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        label="Select Model (for Depth Map)",
                        value=AVAILABLE_MODELS[4] if len(AVAILABLE_MODELS) > 4 else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None) # Default to vitl_fp16
                    )
                    generate_depth_map_button = gr.Button("1. Generate Depth Map", variant="secondary")

                    gr.Markdown("#### SBS 3D Parameters")
                    sbs_method_dropdown = gr.Dropdown(choices=["mesh_warping", "grid_sampling"], value="mesh_warping", label="SBS Method")        
                    sbs_mode_dropdown = gr.Dropdown(choices=["parallel", "cross-eyed"], value="parallel", label="SBS View Mode")
                    sbs_depth_scale_slider = gr.Slider(minimum=1, maximum=150, value=40, step=1, label="SBS Depth Scale")
                    sbs_depth_blur_strength_slider = gr.Slider(minimum=1, maximum=15, value=7, step=2, label="SBS Depth Blur Strength (Odd Values)")
                    generate_sbs_button = gr.Button("2. Generate SBS 3D Image", variant="primary")

            with gr.Row():
                output_sbs_component = gr.Image(type="pil", label="Generated SBS 3D Image", height=480, interactive=False)
        
        with gr.Tab("Video"):
            gr.Markdown("### Video Processing")
            gr.Markdown("Upload a video to process. (Functionality to be implemented)")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input_component = gr.Video(label="Input Video", height=480)
                with gr.Column(scale=1):
                    # Placeholder for video output or controls
                    gr.Markdown("Video output/controls will appear here.")
            # Placeholder for video-specific buttons or parameters
            # generate_video_output_button = gr.Button("Process Video", variant="primary")


    # ========================================
    # IMAGE EVENT HANDLERS
    # ========================================
    # Click handler for generating depth map
    generate_depth_map_button.click(
        fn=generate_depth_map_only,
        inputs=[
            input_image_component, 
            model_dropdown_component
        ],
        outputs=[output_grayscale_component]
    )

    # Click handler for generating SBS image
    generate_sbs_button.click(
        fn=generate_sbs_image_from_depth,
        inputs=[
            input_image_component,         # Original image
            output_grayscale_component,    # Generated depth map
            model_dropdown_component,      # Model name (for dtype)
            sbs_method_dropdown,
            sbs_depth_scale_slider,
            sbs_mode_dropdown,
            sbs_depth_blur_strength_slider
        ],
        outputs=[output_sbs_component]
    )

    # ========================================
    # VIDEO EVENT HANDLERS
    # ========================================
    # needs ffmpeg
    # we need to extract frames from the video (temp dir)
    # produce a depthmap for each from save depth map (temp dir)
    # produce sbs image for each frame, 
    # combine all frames back together to a video (includ audio if original had audio)

if __name__ == "__main__":
    demo.launch()
