import numpy as np
from scipy.io import wavfile
import os

def generate_dummy_audio():
    sample_rate = 16000 # 16 kHz
    duration = 1.0 # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate clean signal (440 Hz sine wave - A4 note)
    freq = 440.0
    clean_signal = np.sin(2 * np.pi * freq * t)
    
    # Add random Gaussian noise
    noise_amplitude = 0.5
    noise = np.random.normal(0, noise_amplitude, clean_signal.shape)
    noisy_signal = clean_signal + noise
    
    # Normalize to 16-bit range
    clean_signal_16bit = np.int16(clean_signal / np.max(np.abs(clean_signal)) * 32767)
    noisy_signal_16bit = np.int16(noisy_signal / np.max(np.abs(noisy_signal)) * 32767)
    
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    wavfile.write(os.path.join(data_dir, 'clean_sample.wav'), sample_rate, clean_signal_16bit)
    wavfile.write(os.path.join(data_dir, 'noisy_sample.wav'), sample_rate, noisy_signal_16bit)
    
    print("Dummy audio samples generated successfully.")

if __name__ == "__main__":
    generate_dummy_audio()
