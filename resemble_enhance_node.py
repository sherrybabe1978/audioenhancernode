import os
import subprocess
import uuid
import folder_paths
from functools import lru_cache

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

    @lru_cache(maxsize=1)
    def _load_enhancer(self, device="cuda", run_dir=None):
        import torch
        from resemble_enhance.enhancer.download import download
        from resemble_enhance.enhancer.hparams import HParams
        from resemble_enhance.enhancer.enhancer import Enhancer

        run_dir = download(run_dir)
        hp = HParams.load(run_dir)
        model = Enhancer(hp)

        ckpt = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(ckpt, map_location="cpu")["module"]
        model.load_state_dict(state_dict)
        model.eval().to(device)
        return model

    def process(self, video_path, mode, solver, nfe, lambd, tau):
        import torch
        import torchaudio
        from resemble_enhance.inference import inference

        input_video = folder_paths.get_annotated_filepath(video_path)
        out_dir = folder_paths.get_output_directory()
        uid = uuid.uuid4().hex[:8]

        raw_audio = os.path.join(out_dir, f"raw_{uid}.wav")
        enhanced_audio = os.path.join(out_dir, f"enhanced_{uid}.wav")
        final_video = os.path.join(out_dir, f"enhanced_video_{uid}.mp4")

        subprocess.run(["ffmpeg", "-y", "-i", input_video, "-vn", "-acodec", "pcm_s16le", raw_audio], check=True)

        dwav, sr = torchaudio.load(raw_audio)
        dwav = dwav.mean(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self._load_enhancer(device=device)
        if mode == "denoise":
            hwav, sr = inference(model=model.denoiser, dwav=dwav, sr=sr, device=device)
        else:
            model.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
            hwav, sr = inference(model=model, dwav=dwav, sr=sr, device=device)

        torchaudio.save(enhanced_audio, hwav[None], sr)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", enhanced_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            final_video
        ], check=True)

        for p in (raw_audio, enhanced_audio):
            if os.path.exists(p):
                os.remove(p)

        return (final_video,)
