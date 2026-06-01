import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Modul untuk mengekspor output berbentuk gambar perbandingan 
    (visualisasi before-after sinyal audio) menggunakan matplotlib.
    """
    
    @staticmethod
    def plot_comparison(original_signal, cleaned_signal, sample_rate, filepath):
        """
        Membuat dan menyimpan grafik subplot atas (original) & bawah (cleaned).
        
        Args:
            original_signal (np.ndarray): Array sinyal audio mentah (ber-noise).
            cleaned_signal (np.ndarray): Array sinyal hasil reduksi noise.
            sample_rate (int): Frekuensi sampling audio.
            filepath (str): Path tujuan penyimpanan gambar (.png/.jpg).
        """
        # Batasi jumlah sampel yang divisualisasikan jika terlalu panjang
        # agar noise bisa terlihat jelas dengan mata telanjang (misal max 1 detik)
        max_samples = min(len(original_signal), sample_rate * 1) 
        
        y_orig = original_signal[:max_samples]
        y_clean = cleaned_signal[:max_samples]
        
        # Sumbu X (waktu)
        time_axis = np.linspace(0, max_samples / sample_rate, num=max_samples)
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Before (Sinyal dengan Noise)
        plt.subplot(2, 1, 1)
        plt.title('Before: Sinyal Audio Asli (dengan Noise)')
        plt.plot(time_axis, y_orig, color='red', alpha=0.7)
        plt.xlabel('Waktu (detik)')
        plt.ylabel('Amplitudo')
        plt.grid(True)
        
        # Subplot 2: After (Sinyal Bersih)
        plt.subplot(2, 1, 2)
        plt.title('After: Sinyal Audio Bersih (Pasca Gradient Descent)')
        plt.plot(time_axis, y_clean, color='blue', alpha=0.7)
        plt.xlabel('Waktu (detik)')
        plt.ylabel('Amplitudo')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Visualisasi perbandingan berhasil disimpan di: {filepath}")

    @staticmethod
    def plot_loss(loss_history, filepath):
        """
        Membuat dan menyimpan grafik kurva penurunan nilai loss/cost.
        
        Args:
            loss_history (list): Riwayat nilai loss tiap epoch.
            filepath (str): Path tujuan penyimpanan gambar.
        """
        plt.figure(figsize=(8, 5))
        plt.title('Kurva Penurunan Cost (Gradient Descent Convergence)')
        plt.plot(loss_history, color='green', linewidth=2)
        plt.xlabel('Epoch (Iterasi)')
        plt.ylabel('Cost Value (Tikhonov Loss)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Visualisasi kurva loss berhasil disimpan di: {filepath}")
