"""Synthetic Audio Dataset Generation for Attribute Binding Evaluation"""

import json
import argparse
from pathlib import Path
from itertools import combinations

import torch
import numpy as np
import librosa
import soundfile as sf
from diffusers import StableAudioPipeline


# Default Configuration
DEFAULT_CONFIG = {
    "num_inference_steps": 200,
    "audio_durations": [5.0, 7.0, 10.0],
    "seed": 0,
    "negative_prompt": "Low quality",
    "model_id": "stabilityai/stable-audio-open-1.0"
}

CATEGORY_CONFIG = {
    "attributes": {
        "Laughter": "laughing",
        "Screaming": "screaming",
        "Crying": "crying",
        "Whispering": "whispering",
        "Singing": "singing",
        "Speech": "speaking",
        "Cough": "coughing",
        "Breathing": "breathing",
    },
    "subjects": ["man", "woman"]
}


def load_pipeline(model_id: str, device: str) -> StableAudioPipeline:
    """Load Stable Audio Pipeline."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def generate_audio(
    pipe: StableAudioPipeline,
    prompt: str,
    seed: int,
    duration: float,
    negative_prompt: str,
    num_inference_steps: int,
    device: str
) -> tuple[np.ndarray, int]:
    """Generate audio from text prompt."""
    generator = torch.Generator(device).manual_seed(seed)
    
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        audio_end_in_s=duration,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios
    
    output = audio[0].T.float().cpu().numpy()
    return output, pipe.vae.sampling_rate


def concatenate_audios(file1: Path, file2: Path, output: Path) -> float:
    """Concatenate two audio files."""
    audio1, sr1 = librosa.load(file1, sr=None)
    audio2, sr2 = librosa.load(file2, sr=None)
    
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
    
    concatenated = np.concatenate([audio1, audio2])
    sf.write(output, concatenated, sr1)
    
    return len(concatenated) / sr1


def generate_samples(pipe, output_dir: Path, config: dict) -> dict:
    """Generate individual audio samples for all subject-attribute combinations."""
    samples = {subj: {attr: {} for attr in CATEGORY_CONFIG["attributes"]} 
               for subj in CATEGORY_CONFIG["subjects"]}
    
    device = next(pipe.unet.parameters()).device.type
    
    total = len(CATEGORY_CONFIG["subjects"]) * len(CATEGORY_CONFIG["attributes"]) * len(config["audio_durations"])
    count = 0
    
    for duration in config["audio_durations"]:
        for subject in CATEGORY_CONFIG["subjects"]:
            for attr_name, attr_prompt in CATEGORY_CONFIG["attributes"].items():
                prompt = f"The sound of a {subject} {attr_prompt}"
                
                print(f"[{count+1}/{total}] Generating: {prompt} (duration={duration}s)")
                
                audio, sr = generate_audio(
                    pipe, prompt, config["seed"], duration,
                    config["negative_prompt"], config["num_inference_steps"], device
                )
                
                filename = f"{subject}_{attr_name}_dur{int(duration)}s.wav"
                sf.write(output_dir / filename, audio, sr)
                
                samples[subject][attr_name][duration] = {
                    "file": filename,
                    "duration": len(audio) / sr,
                    "prompt": prompt
                }
                count += 1
    
    return samples


def create_swap_pairs(samples: dict, audio_dir: Path, swap_dir: Path, durations: list) -> list:
    """Create original and swapped audio pairs."""
    swap_pairs = []
    attributes = list(CATEGORY_CONFIG["attributes"].keys())
    
    pair_id = 0
    for attr1, attr2 in combinations(attributes, 2):
        for dur1 in durations:
            for dur2 in durations:
                man_attr1 = samples["man"][attr1].get(dur1)
                man_attr2 = samples["man"][attr2].get(dur2)
                woman_attr1 = samples["woman"][attr1].get(dur1)
                woman_attr2 = samples["woman"][attr2].get(dur2)
                
                if not all([man_attr1, man_attr2, woman_attr1, woman_attr2]):
                    continue
                
                # Original: Man-attr1 + Woman-attr2
                orig_file = f"pair_{pair_id:04d}_orig_{attr1}_{attr2}.wav"
                orig_duration = concatenate_audios(
                    audio_dir / man_attr1["file"],
                    audio_dir / woman_attr2["file"],
                    audio_dir / orig_file
                )
                
                # Swapped: Woman-attr1 + Man-attr2
                swap_file = f"pair_{pair_id:04d}_swap_{attr1}_{attr2}.wav"
                swap_duration = concatenate_audios(
                    audio_dir / woman_attr1["file"],
                    audio_dir / man_attr2["file"],
                    swap_dir / swap_file
                )
                
                swap_pairs.append({
                    "pair_id": pair_id,
                    "attributes": [attr1, attr2],
                    "original": {
                        "file": orig_file,
                        "duration": orig_duration,
                        "gt": [["man", attr1.lower()], ["woman", attr2.lower()]]
                    },
                    "swapped": {
                        "file": swap_file,
                        "duration": swap_duration,
                        "gt": [["woman", attr1.lower()], ["man", attr2.lower()]]
                    }
                })
                pair_id += 1
                
                if pair_id % 50 == 0:
                    print(f"  Created {pair_id} pairs...")
    
    return swap_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio swap dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_steps", type=int, default=200, help="Inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    swap_dir = output_dir / "swapped"
    audio_dir.mkdir(parents=True, exist_ok=True)
    swap_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = DEFAULT_CONFIG.copy()
    config["num_inference_steps"] = args.num_steps
    config["seed"] = args.seed
    
    print("=" * 60)
    print("Synthetic Audio Dataset Generation")
    print("=" * 60)
    
    # Generate
    pipe = load_pipeline(config["model_id"], args.device)
    
    print("\n[1/3] Generating individual samples...")
    samples = generate_samples(pipe, audio_dir, config)
    
    print("\n[2/3] Creating swap pairs...")
    swap_pairs = create_swap_pairs(samples, audio_dir, swap_dir, config["audio_durations"])
    
    print(f"\n[3/3] Saving metadata...")
    metadata = {
        "config": config,
        "num_pairs": len(swap_pairs),
        "pairs": swap_pairs
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nComplete! Generated {len(swap_pairs)} pairs → {output_dir}")


if __name__ == "__main__":
    main()
