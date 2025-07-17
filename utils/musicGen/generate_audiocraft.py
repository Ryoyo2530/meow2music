import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(meow_path, output_path, prompt="dreamy lofi with cat singing"):
    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=10)

    # Load and resample meow
    waveform, sr = torchaudio.load(meow_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # Use meow as melody
    output = model.generate_with_chroma(
        descriptions=[prompt],
        melody_waveform=waveform,
        melody_sample_rate=sr
    )

    # Save output
    audio_write(output_path, output[0].cpu(), sr=32000)
    print(f"Music saved to {output_path}.wav")

if __name__ == "__main__":
    generate_music("data/meow.wav", "outputs/cat_song")
