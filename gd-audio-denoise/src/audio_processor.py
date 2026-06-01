import numpy as np
from scipy.io import wavfile

class AudioProcessor:
    """
    Kelas untuk menangani pembacaan, penulisan, dan prapemrosesan file audio.
    Sesuai ketentuan, hanya menggunakan scipy.io.wavfile untuk I/O dan numpy untuk komputasi dasar.
    """
    
    @staticmethod
    def load_audio(filepath):
        """
        Membaca file audio .wav.
        
        Args:
            filepath (str): Path ke file .wav.
            
        Returns:
            tuple: (sample_rate, data_audio_ternormalisasi)
        """
        try:
            sample_rate, data = wavfile.read(filepath)
            
            # Jika stereo, konversi ke mono dengan mengambil rata-rata channel
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = data.mean(axis=1)
                
            # Normalisasi data ke rentang [-1.0, 1.0] untuk memudahkan optimasi gradient descent
            # Tipe data biasanya int16 (16-bit PCM), nilai maksimum absolut adalah 32768
            if data.dtype == np.int16:
                data_normalized = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                data_normalized = data.astype(np.float64) / 2147483648.0
            else:
                data_normalized = data.astype(np.float64)
                max_val = np.max(np.abs(data_normalized))
                if max_val > 0:
                    data_normalized = data_normalized / max_val
                    
            return sample_rate, data_normalized
            
        except Exception as e:
            print(f"Error saat memuat file audio: {e}")
            return None, None

    @staticmethod
    def save_audio(filepath, sample_rate, data_normalized):
        """
        Menyimpan array data audio kembali ke file .wav (format 16-bit PCM).
        
        Args:
            filepath (str): Path tujuan file .wav.
            sample_rate (int): Frekuensi sampling (misal 16000 atau 44100).
            data_normalized (np.ndarray): Data audio dalam rentang [-1.0, 1.0].
        """
        try:
            # Denormalisasi kembali ke int16
            # Clipping untuk memastikan tidak melebihi rentang int16
            data_clipped = np.clip(data_normalized, -1.0, 1.0)
            data_int16 = np.int16(data_clipped * 32767.0)
            
            wavfile.write(filepath, sample_rate, data_int16)
        except Exception as e:
            print(f"Error saat menyimpan file audio: {e}")
