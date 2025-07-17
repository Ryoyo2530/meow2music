from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
import torch
import librosa
import soundfile as sf

def generate_music(input_wav = "data/cash_meow.wav",
                   output_wav = "outputs/cat_song.wav"
                   ):
    # ~/.cache/huggingface/hub/models--facebook--musicgen-melody/
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
    
    # Load and resample the meow audio
    wav, sr = librosa.load(input_wav, sr=32000)
    # take 10s
    wav = wav[: sr * 10]
    
    # pass the audio signal directly without using Demucs
    inputs = processor(
        audio=wav,
        sampling_rate=32000,
        text=["dreamy lofi with cat singing"],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=256,
    )

    # Save output
    sampling_rate = model.config.audio_encoder.sampling_rate
    sf.write(output_wav, audio_values[0].T.numpy(), sampling_rate)

if __name__ == "__main__":
    generate_music()
    