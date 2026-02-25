import os
import subprocess
import uuid
import torch
import torchaudio
import folder_paths
from resemble_enhance.enhancer.inference import enhance, denoise

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
    CATEGORY = "AudioPostProduction"

    def process(self, video_path, mode, solver, nfe, lambd, tau):
        input_video = folder_paths.get_annotated_filepath(video_path)
        out_dir = folder_paths.get_output_directory()
        uid = uuid.uuid4().hex[:8]

        raw_audio = os.path.join(out_dir, f"raw_{uid}.wav")
        enhanced_audio = os.path.join(out_dir, f"enhanced_{uid}.wav")
        final_video = os.path.join(out_dir, f"enhanced_video_{uid}.mp4")

        # 1) Extract audio
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_video, "-vn", "-acodec", "pcm_s16le", raw_audio],
            check=True
        )

        # 2) Enhance
        dwav, sr = torchaudio.load(raw_audio)
        dwav = dwav.mean(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if mode == "denoise":
            hwav, sr = denoise(dwav=dwav, sr=sr, device=device)
        else:
            hwav, sr = enhance(dwav=dwav, sr=sr, device=device,
                               solver=solver, nfe=nfe, lambd=lambd, tau=tau)

        torchaudio.save(enhanced_audio, hwav[None], sr)

        # 3) Remux (preserve video timestamps)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", enhanced_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            final_video
        ], check=True)

        # Cleanup
        for p in (raw_audio, enhanced_audio):
            if os.path.exists(p):
                os.remove(p)

        return (final_video,)
