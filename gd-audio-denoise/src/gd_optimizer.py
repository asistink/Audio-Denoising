import numpy as np
import time

class GradientDescentSmoother:
    """
    Mengimplementasikan algoritma Gradient Descent murni (from scratch) 
    untuk menghaluskan sinyal audio.
    
    Fungsi Biaya (Cost Function) yang diminimalkan:
    C(X) = 0.5 * sum((X - Y)^2) + 0.5 * lambda * sum((X[i+1] - X[i])^2)
    Dimana:
    - X adalah sinyal estimasi (bersih)
    - Y adalah sinyal observasi (ber-noise)
    - lambda adalah parameter regularisasi penghalusan (smoothness)
    """
    
    def __init__(self, learning_rate=0.1, max_epochs=100, lambda_reg=5.0, tol=1e-5):
        """
        Inisialisasi hyperparameter algoritma.
        
        Args:
            learning_rate (float): Ukuran langkah (alpha) per iterasi.
            max_epochs (int): Jumlah iterasi maksimal.
            lambda_reg (float): Kekuatan pinalti terhadap nilai turunan.
            tol (float): Batas toleransi perubahan cost untuk early stopping.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.loss_history = []
        
    def _compute_gradient(self, X, Y):
        """
        Menghitung nilai gradien secara manual (from scratch) menggunakan
        operasi array dasar.
        
        Rumus Gradien parsial terhadap X_i:
        - dC/dX_0 = (X_0 - Y_0) + lambda * (X_0 - X_1)
        - dC/dX_i = (X_i - Y_i) + lambda * (2*X_i - X_{i-1} - X_{i+1})
        - dC/dX_{N-1} = (X_{N-1} - Y_{N-1}) + lambda * (X_{N-1} - X_{N-2})
        """
        # Gradien dari term data fitting: (X - Y)
        grad = X - Y
        
        # Tambahan gradien dari term regularisasi
        # Bagian tengah sinyal
        grad[1:-1] += self.lambda_reg * (2 * X[1:-1] - X[:-2] - X[2:])
        
        # Ujung kiri sinyal (awal)
        grad[0] += self.lambda_reg * (X[0] - X[1])
        
        # Ujung kanan sinyal (akhir)
        grad[-1] += self.lambda_reg * (X[-1] - X[-2])
        
        return grad

    def _compute_cost(self, X, Y):
        """
        Menghitung nilai fungsi biaya (cost) untuk memantau proses optimasi.
        """
        data_fidelity = 0.5 * np.sum((X - Y)**2)
        regularization = 0.5 * self.lambda_reg * np.sum((X[1:] - X[:-1])**2)
        return data_fidelity + regularization

    def denoise(self, noisy_signal):
        """
        Menjalankan proses optimasi Gradient Descent.
        
        Args:
            noisy_signal (np.ndarray): Sinyal input yang ber-noise (1D array).
            
        Returns:
            np.ndarray: Sinyal yang telah dihaluskan.
        """
        Y = noisy_signal
        
        # Inisialisasi tebakan awal X dengan sinyal kotor itu sendiri
        X = np.copy(Y)
        self.loss_history = []
        
        print(f"Memulai iterasi Gradient Descent (maks {self.max_epochs} epochs)...")
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # 1. Hitung gradien
            grad = self.compute_gradient(X, Y)
            
            # 2. Update sinyal X bergerak berlawanan arah gradien
            X = X - (self.learning_rate * grad)
            
            # 3. Hitung cost untuk early stopping dan monitoring
            cost = self.compute_cost(X, Y)
            self.loss_history.append(cost)
            
            if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                print(f"Epoch {epoch:4d} | Cost: {cost:.4f}")
                
            # Early stopping jika perubahan cost sangat kecil
            if epoch > 0 and abs(self.loss_history[-2] - cost) < self.tol:
                print(f"Konvergensi tercapai pada epoch {epoch}.")
                break
                
        elapsed = time.time() - start_time
        print(f"Proses optimasi selesai dalam {elapsed:.2f} detik.")
        
        return X
    
    # Exposing the private methods slightly for testing if needed
    compute_gradient = _compute_gradient
    compute_cost = _compute_cost
