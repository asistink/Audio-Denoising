import os
import argparse
from audio_processor import AudioProcessor
from gd_optimizer import GradientDescentSmoother
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Gradient Descent Audio Denoising")
    parser.add_argument("--input", type=str, default="../data/noisy_sample.wav", help="Path ke file audio input (.wav)")
    parser.add_argument("--output_audio", type=str, default="../data/cleaned_sample.wav", help="Path keluaran file audio bersih")
    parser.add_argument("--epochs", type=int, default=500, help="Jumlah maksimum iterasi Gradient Descent")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (alpha)")
    parser.add_argument("--lam", type=float, default=20.0, help="Faktor regularisasi Tikhonov (semakin tinggi semakin halus)")
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input))
    output_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_audio))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
    
    print(f"--- Gradient Descent Audio Denoising ---")
    print(f"Membaca file: {input_path}")
    
    # 1. Load Audio
    sample_rate, noisy_signal = AudioProcessor.load_audio(input_path)
    
    if noisy_signal is None:
        print("Gagal membaca audio. Keluar program.")
        return
        
    print(f"Audio termuat. Sample Rate: {sample_rate} Hz, Durasi: {len(noisy_signal)/sample_rate:.2f} detik.")
    
    # 2. Proses Denoising (Murni Matematika from scratch)
    smoother = GradientDescentSmoother(
        learning_rate=args.lr, 
        max_epochs=args.epochs, 
        lambda_reg=args.lam,
        tol=1e-6
    )
    
    cleaned_signal = smoother.denoise(noisy_signal)
    
    # 3. Simpan Audio Bersih
    print(f"Menyimpan audio bersih ke: {output_audio_path}")
    AudioProcessor.save_audio(output_audio_path, sample_rate, cleaned_signal)
    
    # 4. Visualisasi (Output Gambar)
    print("Mengekspor gambar perbandingan visual...")
    plot_path = os.path.join(output_dir, "denoising_comparison.png")
    Visualizer.plot_comparison(noisy_signal, cleaned_signal, sample_rate, plot_path)
    
    loss_path = os.path.join(output_dir, "loss_curve.png")
    Visualizer.plot_loss(smoother.loss_history, loss_path)
    
    print("Proses selesai dengan sukses!")

if __name__ == "__main__":
    main()
