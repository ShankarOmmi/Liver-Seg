import gradio as gr
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from monai.networks.nets import UNet

# -----------------------------
# 1️⃣ DEVICE
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# 2️⃣ LOAD MODEL (Same as Training)
# -----------------------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# -----------------------------
# 3️⃣ SEGMENT FUNCTION
# -----------------------------
def segment_middle_slice(nii_file):

    # Load NIfTI
    volume = nib.load(nii_file.name).get_fdata()

    # Get middle slice (axial)
    middle_index = volume.shape[2] // 2
    slice_img = volume[:, :, middle_index]

    # Transpose same as training
    slice_img = slice_img.transpose(1, 0)

    # -------------------------
    # SAME PREPROCESSING AS TRAINING
    # -------------------------
    min_val, max_val = -30, 140
    slice_img = np.clip(slice_img, min_val, max_val)
    slice_img = (slice_img - min_val) / (max_val - min_val)

    # Resize to 256x256
    pil_img = Image.fromarray((slice_img * 255).astype(np.uint8))
    pil_img = pil_img.resize((256, 256))
    slice_img = np.array(pil_img).astype(np.float32) / 255.0

    # Prepare tensor (B, C, H, W)
    slice_img = np.expand_dims(slice_img, axis=0)
    slice_img = np.expand_dims(slice_img, axis=0)

    input_tensor = torch.tensor(slice_img).float().to(device)

    # -------------------------
    # INFERENCE
    # -------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    # -------------------------
    # CREATE OVERLAY (Professional Look)
    # -------------------------
    original = (slice_img.squeeze() * 255).astype(np.uint8)
    overlay = np.stack([original]*3, axis=-1)

    overlay[mask == 1] = [255, 0, 0]  # Red mask

    return overlay

# -----------------------------
# 4️⃣ GRADIO UI
# -----------------------------
interface = gr.Interface(
    fn=segment_middle_slice,
    inputs=gr.File(label="Upload NIfTI (.nii) volume"),
    outputs=gr.Image(type="numpy"),
    title="Liver Segmentation (MONAI UNet)",
    description="Upload a CT volume (.nii). The model segments the middle slice."
)

interface.launch()