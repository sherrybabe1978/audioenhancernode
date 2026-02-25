# ============================================================
# DEEPSPEED MOCK — must be first, before resemble_enhance
# ============================================================
import sys
from unittest.mock import MagicMock

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
    OUTPUT_NODE = True          # ← THIS is the key fix! Marks it as a terminal node
    CATEGORY = "audio"

    def process(self, video_path, mode, solver, nfe, lambd, tau):
        from resemble_enhance.enhancer.inference import enhance, denoise

        # ── Resolve input path ──────────────────────────────────────────
        # Handle full paths, relative paths, or just filenames
        if os.path.isabs(video_path) and os.path.exists(video_path):
            input_video = video_path
        else:
            # Strip any leading "input/" prefix
            filename = os.path.basename(video_path)
            input_dir = folder_paths.get_input_directory()
            input_video = os.path.join(input_dir, filename)

        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Video not found: {input_video}")

        out_dir = folder_paths.get_output_directory()
        uid = uuid.uuid4().hex[:8]

        raw_audio    = os.path.join(out_dir, f"raw_{uid}.wav")
        enhanced_audio = os.path.join(out_dir, f"enhanced_{uid}.wav")
        final_video  = os.path.join(out_dir, f"enhanced_video_{uid}.mp4")

        # ── Extract audio ───────────────────────────────────────────────
        subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-vn", "-acodec", "pcm_s16le", raw_audio
        ], check=True)

        dwav, sr = torchaudio.load(raw_audio)
        dwav = dwav.mean(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Enhance ─────────────────────────────────────────────────────
        if mode == "denoise":
            hwav, sr_out = denoise(dwav, sr, device)
        else:
            hwav, sr_out = enhance(
                dwav, sr, device,
                nfe=nfe, solver=solver, lambd=lambd, tau=tau
            )

        torchaudio.save(enhanced_audio, hwav.unsqueeze(0), sr_out)

        # ── Merge back into video ───────────────────────────────────────
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", enhanced_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            final_video
        ], check=True)

        # ── Cleanup ─────────────────────────────────────────────────────
        for p in (raw_audio, enhanced_audio):
            if os.path.exists(p):
                os.remove(p)

        print(f"✅ Enhanced video saved to: {final_video}")
        return (final_video,)


NODE_CLASS_MAPPINGS = {
    "ResembleVideoAudioEnhancer": ResembleVideoAudioEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResembleVideoAudioEnhancer": "🎧 Resemble Video Audio Enhancer"
}
