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
import cv2
import tempfile
import shutil
import imageio
import subprocess

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
    'depth_anything_v2_vitl_fp32.safetensors'
]

# AVAILABLE_MODELS = [
#     'depth_anything_v2_vitl_fp16.safetensors',    
#     'depth_anything_v2_vitb_fp16.safetensors',
#     'depth_anything_v2_vits_fp16.safetensors'        
# ]


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

def process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base, output_dir_frames="output"): # Added output_dir_frames with default for backward compatibility if called elsewhere
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
        # print(f"Peak GPU Memory Allocated during inference: {peak_memory_mib:.2f} MiB ({peak_memory_gib:.2f} GiB)")

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
    # output_dir = "output" # Old hardcoded output
    os.makedirs(output_dir_frames, exist_ok=True) # Use new parameter
    
    # Save grayscale depth map
    grayscale_path = os.path.join(output_dir_frames, f"{output_filename_base}_depth.png")
    depth_image.save(grayscale_path)
    print(f"Saved grayscale depth map to: {grayscale_path}")
    
    # Save colored depth map
    colored_path = os.path.join(output_dir_frames, f"{output_filename_base}_depth_colored.png") # Use new parameter
    # colored_depth_image.save(colored_path)
    # print(f"Saved colored depth map to: {colored_path}")
    
    return depth_image

def generate_depth_map_only(input_image, model_name):
    """
    Generates only the depth map from the input image.
    called by generate_depth_and_sbs_combined()
    """
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
            gr.Info(f"SBS Depth Blur Strength adjusted to {sbs_depth_blur_strength} (must be odd-numbered).")

        # print(f"Calling process_image_sbs with method: {sbs_method}, scale: {sbs_depth_scale}, mode: {sbs_mode}, blur: {sbs_depth_blur_strength}")
        sbs_image_tensor = process_image_sbs(
                base_image=base_image_for_sbs,
                depth_map=depth_map_for_sbs,
                method=sbs_method,
                depth_scale=sbs_depth_scale,
                mode=sbs_mode,
                depth_blur_strength=sbs_depth_blur_strength
            )
        
        # print(f"[run_gradio.generate_sbs_from_depth] sbs_image_tensor shape: {sbs_image_tensor.shape}", color.YELLOW)
        sbs_image_pil = transforms.ToPILImage()(sbs_image_tensor.squeeze(0).cpu().permute(2, 0, 1))
        print("SBS image generated successfully.")
        return sbs_image_pil
    
    except Exception as e:
        print(f"Error generating SBS image: {e}")
        gr.Error(f"Error generating SBS image: {e}")
        return None


def generate_depth_and_sbs_combined(input_image, model_name, sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength):
    """Combined function that generates depth map and then SBS image in sequence."""
    
    # Step 1: Generate depth map
    print("Step 1: Generating depth map...")
    depth_map = generate_depth_map_only(input_image, model_name)
    
    if depth_map is None:
        return None, None  # Return None for both outputs if depth map generation fails
    
    # Step 2: Generate SBS image using the generated depth map
    print("Step 2: Generating SBS 3D image...")
    sbs_image = generate_sbs_image_from_depth(
        input_image, 
        depth_map, 
        model_name, 
        sbs_method, 
        sbs_depth_scale, 
        sbs_mode, 
        sbs_depth_blur_strength
    )
    
    return depth_map, sbs_image  # Return both outputs

def generate_sbs_video(video_path, model_name, sbs_method, sbs_mode, sbs_depth_scale, sbs_depth_blur_strength, progress=gr.Progress(track_tqdm=True)):
    if not video_path:
        gr.Warning("Please upload a video to process.")
        return None

    # 1. Setup (device, dtype, load depth model)
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")
    print(f"Using device: {device} for video processing.")
    
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    
    try:
        depth_model, _, is_metric = load_model(model_name, device) # Unpack model, dtype (ignore), is_metric
    except Exception as e:
        gr.Error(f"Failed to load model: {e}")
        return None

    # 2. Create Temporary Directories
    temp_parent_dir = tempfile.mkdtemp(prefix="sbs_video_")
    frames_orig_dir = os.path.join(temp_parent_dir, "frames_orig")
    frames_depth_dir = os.path.join(temp_parent_dir, "frames_depth")
    frames_sbs_dir = os.path.join(temp_parent_dir, "frames_sbs")
    os.makedirs(frames_orig_dir, exist_ok=True)
    os.makedirs(frames_depth_dir, exist_ok=True)
    os.makedirs(frames_sbs_dir, exist_ok=True)

    output_video_base_name = f"sbs_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    final_output_video_path = os.path.join("output", output_video_base_name) # Ensure "output" dir exists
    os.makedirs("output", exist_ok=True)

    try:
        # 3. Video Info & Audio Extraction
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: 
            gr.Warning("Could not determine video FPS. Defaulting to 25. Output video might have incorrect speed.")
            fps = 25.0 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        temp_audio_path = os.path.join(temp_parent_dir, "audio.aac")
        audio_extracted = False
        try:
            ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
            probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
            if probe_result.returncode == 0 and probe_result.stdout.strip():
                cmd_extract_audio = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', temp_audio_path]
                subprocess.run(cmd_extract_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio_extracted = True
                print("Audio extracted successfully.")
            else:
                print(f"No audio stream found or ffprobe error. Probe output: {probe_result.stdout.strip()} {probe_result.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error during audio extraction: {e.stderr.decode() if e.stderr else e.stdout.decode() if e.stdout else 'Unknown error'}")
        except FileNotFoundError:
            print("ffmpeg/ffprobe not found. Audio will not be processed. Please ensure ffmpeg is installed and in PATH.")

        # 4. Frame Extraction
        print(f"Extracting {frame_count} frames at {fps} FPS...")
        actual_frames_extracted = 0
        for i in progress.tqdm(range(frame_count), desc="Extracting Frames"):
            ret, frame = cap.read()
            if not ret: 
                print(f"Warning: Could only read {i} frames out of {frame_count}.")
                frame_count = i # Adjust frame count if video ends prematurely
                break
            cv2.imwrite(os.path.join(frames_orig_dir, f"frame_{i:06d}.png"), frame)
            actual_frames_extracted += 1
        cap.release()

        if actual_frames_extracted == 0: # Use actual_frames_extracted
            gr.Error("No frames could be extracted from the video.")
            return None

        # 5. Process Each Frame
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        orig_frame_files = sorted([f for f in os.listdir(frames_orig_dir) if f.endswith(".png")])
        for frame_idx, frame_filename in enumerate(progress.tqdm(orig_frame_files, desc="Processing Frames")):
            frame_path = os.path.join(frames_orig_dir, frame_filename)
            base_name = os.path.splitext(frame_filename)[0]
            
            input_pil_image = Image.open(frame_path).convert("RGB")
            
            # Depth Map
            image_tensor = transform_normalize(input_pil_image).unsqueeze(0).to(device=device, dtype=dtype)
            # Call modified process_depthmap_image, providing the specific output directory for depth frames
            depth_pil_image = process_depthmap_image(depth_model, image_tensor, device, dtype, is_metric, base_name, frames_depth_dir) 
            
            # SBS Image
            sbs_pil_image = generate_sbs_image_from_depth(
                input_pil_image, depth_pil_image, model_name, 
                sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength
            )
            if sbs_pil_image:
                sbs_pil_image.save(os.path.join(frames_sbs_dir, f"sbs_{base_name}.png"))
            else:
                gr.Warning(f"Failed to generate SBS for frame {frame_filename}. Skipping.")

        # 6. Assemble SBS Video
        sbs_frame_files = sorted([os.path.join(frames_sbs_dir, f) for f in os.listdir(frames_sbs_dir) if f.startswith("sbs_") and f.endswith(".png")])
        
        if not sbs_frame_files:
            gr.Error("No SBS frames were generated. Cannot create video.")
            return None

        sbs_video_no_audio_path = os.path.join(temp_parent_dir, "sbs_video_no_audio.mp4")
        
        print(f"Assembling SBS video from {len(sbs_frame_files)} frames at {fps} FPS...")
        with imageio.get_writer(sbs_video_no_audio_path, fps=fps, codec='libx264', ffmpeg_params=['-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p']) as writer:
            for sbs_frame_file in progress.tqdm(sbs_frame_files, desc="Assembling Video"):
                writer.append_data(imageio.imread(sbs_frame_file))
        
        # 7. Add Audio Back (if extracted)
        if audio_extracted and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            print(f"Adding audio back to video: {final_output_video_path}")
            cmd_mux = ['ffmpeg', '-y', '-i', sbs_video_no_audio_path, '-i', temp_audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-shortest', final_output_video_path]
            mux_result = subprocess.run(cmd_mux, capture_output=True, text=True, check=False)
            if mux_result.returncode != 0:
                print(f"ffmpeg error during audio muxing: {mux_result.stderr}")
                print("Falling back to video without audio.")
                shutil.move(sbs_video_no_audio_path, final_output_video_path)
            else:
                print("Audio muxed successfully.")
        else:
            print(f"Saving video without audio (or audio processing failed/not present): {final_output_video_path}")
            shutil.move(sbs_video_no_audio_path, final_output_video_path)
        
        print(f"Video processing complete. Output: {final_output_video_path}")
        return final_output_video_path

    except Exception as e:
        gr.Error(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 8. Cleanup
        if os.path.exists(temp_parent_dir):
            print(f"Cleaning up temporary directory: {temp_parent_dir}")
            shutil.rmtree(temp_parent_dir)

# ================================================
# GRADIO UI
with gr.Blocks(title="SBS 2D To 3D") as demo:
    
    gr.Markdown("## SBS 2D To 3D Demo")

    with gr.Tabs():
        with gr.Tab("Image"):
            
            gr.Markdown("Upload an image, generate its depth map, then generate the 3D SBS image.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_component = gr.Image(label="Input Image", type="pil", height=420)
                    
                with gr.Column(scale=1):
                    output_grayscale_component = gr.Image(label="Generated Depth Map", type="pil", height=420, interactive=False) # Depth map is output here
                    
                with gr.Column(scale=1):
                    model_dropdown_component = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        label="Select Model (for Depth Map)",
                        value=AVAILABLE_MODELS[4] if len(AVAILABLE_MODELS) > 4 else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None) # Default to vitl_fp16
                    )
                    # generate_depth_map_button = gr.Button("1. Generate Depth Map", variant="secondary")

                    with gr.Group():
                        gr.Markdown("#### SBS 3D Parameters")
                        with gr.Row():
                            sbs_method_image = gr.Dropdown(choices=["mesh_warping", "grid_sampling"], value="mesh_warping", label="SBS Method")        
                            sbs_mode_image = gr.Dropdown(choices=["parallel", "cross-eyed"], value="parallel", label="SBS View Mode")
                        
                        sbs_depth_scale_image = gr.Slider(minimum=1, maximum=150, value=40, step=1, label="SBS Depth Scale")
                        sbs_depth_blur_strength_image = gr.Slider(minimum=1, maximum=15, value=7, step=2, label="SBS Depth Blur Strength")
                    generate_sbs_button = gr.Button("Generate SBS 3D Image", variant="primary") # UPDATED TEXT

            with gr.Row():
                output_sbs_component = gr.Image(type="pil", label="Generated SBS 3D Image", height=480, interactive=False)
        
        with gr.Tab("Video"):
            gr.Markdown("### Video Processing")
            gr.Markdown("Upload a video to process)")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input_component = gr.Video(label="Input Video", height=480)
                with gr.Column(scale=1):
                    model_dropdown_video = gr.Dropdown( # Renamed
                        choices=AVAILABLE_MODELS,
                        label="Select Model (for Depth Map)",
                        value=AVAILABLE_MODELS[4] if len(AVAILABLE_MODELS) > 4 else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None) # Default to vitl_fp16
                    )
                    gr.Markdown("#### SBS 3D Parameters")
                    sbs_method_video = gr.Dropdown(choices=["mesh_warping", "grid_sampling"], value="mesh_warping", label="SBS Method") # Renamed       
                    sbs_mode_video = gr.Dropdown(choices=["parallel", "cross-eyed"], value="parallel", label="SBS View Mode") # Renamed
                    sbs_depth_scale_video = gr.Slider(minimum=1, maximum=150, value=40, step=1, label="SBS Depth Scale") # Renamed
                    sbs_depth_blur_strength_video = gr.Slider(minimum=1, maximum=15, value=7, step=2, label="SBS Depth Blur Strength (Odd Values)") # Renamed
                    process_video_button = gr.Button("Process SBS 3D Video", variant="primary")


            
            with gr.Row():
                # show result of the merged sbs video
                output_sbs_video_component = gr.Video(label="Generated SBS 3D Video", interactive=False) # Defined

        gr.Markdown(
            """
        | **Parameter** | **Description** |
        |---------------|-----------------|
        | `Model` | Selects the Depth Anything V2 model for depth map generation. Different models (VITS (small), VITB (base), VITL (large)) offer trade-offs in speed and accuracy. <br> "If it does not exist, it will download the selected model into the directory `models/depthanything`. The default `...vitl-fp16` variant is approx 600MB. |
        | `SBS Method` | The algorithm used to generate the Side-by-Side 3D image from the depth map. <br> - `mesh_warping`: Warps the image based on a 3D mesh derived from the depth map. Generally provides good results. <br> - `grid_sampling`: Samples pixels from the original image based on a grid distorted by the depth map. Can be faster but might produce different visual artifacts. |
        | `SBS View Mode` | Determines the arrangement of the left and right eye views in the SBS image. <br> - `parallel`: Left eye view on the left, right eye view on the right. Suitable for parallel viewing. <br> - `cross-eyed`: Right eye view on the left, left eye view on the right. Suitable for cross-eyed viewing. <br><br> - *Cross-eyed mode is primarily used for viewing stereoscopic 3D images on a regular 2D screen without needing any special equipment. <br> With Parallel, the left-eye image feeds the left eye, and the right-eye image feeds the right eye. <br> But with Cross-eyed, this is flipped it places the left-eye image on the right side and the right-eye image on the left side. <br> When you cross your eyes, each eye ends up looking at the correct image, and your brain fuses them into a 3D image which will appear centered. <br> If done correctly, that middle image will appear 3D without the need for a VR headset or 3D glasses.*|
        | `SBS Depth Scale` | Controls the intensity of the 3D effect. Higher values increase the perceived depth and separation between foreground and background objects. Range: 1-150. |
        | `SBS Depth Blur Strength` | Applies a blur to the depth map before SBS generation. This can help smooth out artifacts in the depth map and create a softer 3D effect. Must be an odd number. Range: 1-15. |

        """
        )
    # ========================================
    # IMAGE EVENT HANDLERS
    # ========================================
    # Click handler for generating depth map
    # generate_depth_map_button.click(
    #     fn=generate_depth_map_only,
    #     inputs=[
    #         input_image_component, 
    #         model_dropdown_component
    #     ],
    #     outputs=[output_grayscale_component]
    # )

    # Click handler for generating SBS image
    generate_sbs_button.click(
        fn=generate_depth_and_sbs_combined,
        inputs=[
            input_image_component,         # Original image
            model_dropdown_component,      # Model name
            sbs_method_image,
            sbs_depth_scale_image,
            sbs_mode_image,
            sbs_depth_blur_strength_image
        ],
        outputs=[output_grayscale_component, output_sbs_component]
    )

    # ========================================
    # VIDEO EVENT HANDLERS
    # ========================================


    # uses imageio-ffmpeg
    # we need to extract frames from the video (temp dir)
    # produce a depthmap for each from save depth map (temp dir)
    # produce sbs image for each frame, 
    # combine all frames back together to a video (includ audio if original had audio)
    process_video_button.click(
        fn=generate_sbs_video,
        inputs=[
            video_input_component,
            model_dropdown_video, 
            sbs_method_video,     
            sbs_mode_video,       
            sbs_depth_scale_video,
            sbs_depth_blur_strength_video
        ],
        outputs=[output_sbs_video_component]
    )

if __name__ == "__main__":
    demo.launch()


"""
Image Processing Path
===========================

generate_sbs_button.click()
└── generate_depth_and_sbs_combined()
    ├── generate_depth_map_only()
    │   ├── load_model()
    │   │   ├── hf_hub_download() [if model not cached]
    │   │   ├── load_safetensors()
    │   │   └── DepthAnythingV2() [model initialization]
    │   └── process_depthmap_image()
    │       ├── F.interpolate() [resize to multiple of 14]
    │       ├── model() [inference]
    │       ├── depth normalization & post-processing
    │       └── Image.fromarray() [convert to PIL]
    └── generate_sbs_image_from_depth()
        └── process_image_sbs() [from sbs.sbs module]

Video Processing Path
===========================

process_video_button.click()
└── generate_sbs_video()
    ├── load_model()
    │   ├── hf_hub_download() [if model not cached]
    │   ├── load_safetensors()
    │   └── DepthAnythingV2() [model initialization]
    ├── tempfile.mkdtemp() [create temp directories]
    ├── cv2.VideoCapture() [video info extraction]
    ├── subprocess.run() [ffmpeg audio extraction]
    ├── Frame Extraction Loop:
    │   └── cv2.imwrite() [save each frame as PNG]
    ├── Frame Processing Loop (for each frame):
    │   ├── process_depthmap_image()
    │   │   ├── F.interpolate() [resize to multiple of 14]
    │   │   ├── model() [inference]
    │   │   ├── depth normalization & post-processing
    │   │   └── Image.fromarray() [convert to PIL]
    │   └── generate_sbs_image_from_depth()
    │       └── process_image_sbs() [from sbs.sbs module]
    ├── imageio.get_writer() [assemble video from SBS frames]
    ├── subprocess.run() [ffmpeg audio muxing]
    └── shutil.rmtree() [cleanup temp directories]
"""
