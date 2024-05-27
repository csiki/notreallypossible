import numpy as np
import wave


def read_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        frames = wav_file.readframes(n_frames)
        dtype = np.int16 if sampwidth == 2 else np.int8

        audio_data = np.frombuffer(frames, dtype=dtype)

        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)

        return audio_data, framerate


def save_wav(data, out_path, sampling_rate):
    # Open a WAV file in write mode
    with wave.open(out_path, 'w') as wav_file:
        # Set parameters
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample, since uint16 is 16 bits
        wav_file.setframerate(sampling_rate)

        # Convert numpy array to bytes and write to WAV file
        wav_file.writeframes(data.tobytes())
