# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from typing import List
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

# Model specific imports
import torchaudio

AUDIO_CACHE = 'checkpoints'
AUDIO_URL = "https://weights.replicate.delivery/default/facebookresearch/audiocraft/magnet.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # set the env variable AUDIOCRAFT_CACHE_DIR
        os.environ['AUDIOCRAFT_CACHE_DIR'] = AUDIO_CACHE
        if not os.path.exists(AUDIO_CACHE):
            download_weights(AUDIO_URL, AUDIO_CACHE)
        self.model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input Text",
            default="80s electronic track with melodic synthesizers, catchy beat and groovy bass"
        ),
        input_audio: Path = Input(
            description="An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody.",
            default=None,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue from `input_audio`. Otherwise, generated music will mimic `input_audio`'s melody.",
            default=False,
        ),
        continuation_start: int = Input(
            description="Start time of the audio file to use for continuation.",
            default=0,
            ge=0,
        ),
        continuation_end: int = Input(
            description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        model: str = Input(
            description="Model to use",
            default="facebook/magnet-small-10secs",
            choices=[
                'facebook/magnet-small-10secs',
                'facebook/magnet-medium-10secs',
                'facebook/magnet-small-30secs',
                'facebook/magnet-medium-30secs',
                'facebook/audio-magnet-small',
                'facebook/audio-magnet-medium']
        ),
        variations: int = Input(
            description="Number of variations to generate",
            default=3, ge=1, le=4,
        ),
        span_score: str = Input(
            default="prod-stride1",
            choices=["max-nonoverlap", "prod-stride1"],
        ),
        temperature: float = Input(
            default=3.0,
            description="Temperature for sampling",
        ),
        top_p: float = Input(
            default=0.9, ge=0.0, le=1.0,
            description="Top p for sampling",
        ),
        max_cfg: float = Input(
            default=10.0,
            description="Max CFG coefficient",
        ),
        min_cfg: float = Input(
            default=1.0,
            description="Min CFG coefficient",
        ),
        decoding_steps_stage_1: int = Input(
            default=20,
            description="Number of decoding steps for stage 1",
        ),
        decoding_steps_stage_2: int = Input(
            default=10,
            description="Number of decoding steps for stage 2",
        ),
        decoding_steps_stage_3: int = Input(
            default=10,
            description="Number of decoding steps for stage 3",
        ),
        decoding_steps_stage_4: int = Input(
            default=10,
            description="Number of decoding steps for stage 4",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        descriptions = [prompt for _ in range(variations)]

        self.model = MAGNeT.get_pretrained(model)

        if continuation and input_audio:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            self.model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_p=top_p,
                max_cfg_coef=max_cfg, min_cfg_coef=min_cfg, 
                decoding_steps=[decoding_steps_stage_1, decoding_steps_stage_2, decoding_steps_stage_3, decoding_steps_stage_4],
                span_arrangement='stride1' if (span_score == 'prod-stride1') else 'nonoverlap',)
            wav, tokens = model.generate_continuation(
                prompt=input_audio_wavform,
                prompt_sample_rate=sr,
                descriptions=[prompt],
                progress=True,
                return_tokens=True
            )

        else:
            self.model.set_generation_params(
                duration=duration,
                temperature=temperature,
                top_p=top_p,
                max_cfg_coef=max_cfg, min_cfg_coef=min_cfg, 
                decoding_steps=[decoding_steps_stage_1, decoding_steps_stage_2, decoding_steps_stage_3, decoding_steps_stage_4],
                span_arrangement='stride1' if (span_score == 'prod-stride1') else 'nonoverlap',)
            wav = self.model.generate(descriptions)

        #Delete older runs
        os.system("rm -rf /tmp/output")

        for idx, one_wav in enumerate(wav):
            audio_write(f'/tmp/output/{idx}', one_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)

        output_paths = []
        for idx in range(variations):
            output_paths.append(Path(f'/tmp/output/{idx}.wav'))

        return output_paths
