# ============================================================
# DEEPSPEED MOCK — must be first, before resemble_enhance
# resemble_enhance/enhancer/train.py hardcodes:
#   from deepspeed import DeepSpeedConfig
# We don't need training, just inference — so we mock it out.
# ============================================================
import sys
from unittest.mock import MagicMock

# Create a mock deepspeed module with the attributes train.py needs
_mock_deepspeed = MagicMock()
_mock_deepspeed.DeepSpeedConfig = MagicMock()
_mock_deepspeed.DeepSpeedEngine = MagicMock()
_mock_deepspeed.initialize = MagicMock()
sys.modules['deepspeed'] = _mock_deepspeed
sys.modules['deepspeed.runtime'] = MagicMock()
sys.modules['deepspeed.runtime.config'] = MagicMock()
sys.modules['deepspeed.ops'] = MagicMock()
sys.modules['deepspeed.ops.adam'] = MagicMock()
# ============================================================

import os
import subprocess
import uuid
import folder_paths
import torch
import torchaudio

class ResembleVideoAudioEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input/video.mp4"}),
                "mode": (["enhance", "denoise"],),
                "solver": (["midpoint", "rk4", "euler"],),
                "nfe": ("INT", {"default": 64, "min": 1, "max": 128}),
                "lambd": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_video_path",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, video_path, mode, solver, nfe, lambd, tau):
        # Import here (lazy) so mock is already in place
        from resemble_enhance.enhancer.inference import enhance, denoise

        input_video = folder_paths.get_annotated_filepath(video_path)
        out_dir = folder_paths.get_output_directory()
        uid = uuid.uuid4().hex[:8]

        raw_audio = os.path.join(out_dir, f"raw_{uid}.wav")
        enhanced_audio = os.path.join(out_dir, f"enhanced_{uid}.wav")
        final_video = os.path.join(out_dir, f"enhanced_video_{uid}.mp4")

        # Extract audio from video
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-vn", "-acodec", "pcm_s16le", raw_audio
        ], check=True)

        dwav, sr = torchaudio.load(raw_audio)
        dwav = dwav.mean(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Run enhancement
        if mode == "denoise":
            hwav, sr_out = denoise(dwav, sr, device)
        else:
            hwav, sr_out = enhance(
                dwav, sr, device,
                nfe=nfe, solver=solver, lambd=lambd, tau=tau
            )

        torchaudio.save(enhanced_audio, hwav.unsqueeze(0), sr_out)

        # Merge enhanced audio back into video
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", enhanced_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            final_video
        ], check=True)

        # Cleanup temp files
        for p in (raw_audio, enhanced_audio):
            if os.path.exists(p):
                os.remove(p)

        return (final_video,)


NODE_CLASS_MAPPINGS = {
    "ResembleVideoAudioEnhancer": ResembleVideoAudioEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResembleVideoAudioEnhancer": "🎧 Resemble Video Audio Enhancer"
}
