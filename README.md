# Non-ComfyUI version of my SBS 2D-To-3D Conversion scripts.

Work-In-Progress

This project utilizes the Depth Anything V2 model (safetensors formats)
Original [repository](https://github.com/DepthAnything/Depth-Anything-V2).

## Installation

Follow these steps to set up the project environment and install dependencies.

**1. Clone the Repository (Optional)**

If you haven't already, clone the repository to your local machine:

```bash
git clone https://github.com/yushan777/SBS-2DTo3D.git
cd SBS-2DTo3D
```

**2. Create a Python Virtual Environment**

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv venv 
# Or python -m venv venv
```
*   This command creates a directory named `venv` in your project folder containing a dedicated Python installation.

**3. Activate the Virtual Environment**

You need to activate the environment before installing packages. The command differs based on your operating system:

*   **macOS / Linux (bash/zsh):**
    ```bash
    source venv/bin/activate
    ```
*   **Windows (Command Prompt):**
    ```bash
    venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell):**
    ```bash
    venv\Scripts\Activate.ps1
    ```
    *(Note: You might need to adjust your PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

    You'll know the environment is active when you see `(venv)` prepended to your command prompt.

**4. Install PyTorch (Hardware-Specific)**

Run the provided script to detect your hardware and install the correct PyTorch build (including `torchvision` and `torchaudio`):

```bash
python install_torch.py
```
*   This script checks for NVIDIA GPUs (installing the CUDA version if found) or Apple Silicon (installing the standard version with MPS support).

**5. Install Remaining Dependencies**

Install the other required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage Example

To run the model on an image:

```bash
python main.py \
    --input path/to/your/image.jpg \
    ----output-dir path/to/save/depth_maps \
    --model depth_anything_v2_vitl_fp32.safetensors
```
*   Replace `path/to/your/image.jpg` and `path/to/save/depth_maps` with your actual file and dir paths.
*   Choose the desired model using the `--model` argument.
*   The script will automatically attempt to use a GPU (CUDA or MPS) if available, falling back to CPU otherwise.

**6. Deactivate the Virtual Environment (When Done)**

When you're finished working on the project, you can deactivate the environment:

```bash
deactivate
```

You should now have all the necessary dependencies installed to run the project.
