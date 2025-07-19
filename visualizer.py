import matplotlib.pyplot as plt
import os

class SpectrumVisualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_spectra(self, target_spectrum, combined_spectrum):
        plt.figure(figsize=(14, 7))
        plt.plot(target_spectrum["Wavelength"], target_spectrum["Spectral_irradiance"], label="Эталонный спектр")
        plt.plot(combined_spectrum["Wavelength"], combined_spectrum["Spectral_irradiance"], label="Синтезированный спектр", linestyle="--")
        plt.title("Сравнение эталонного и синтезированного спектров")
        plt.xlabel("Длина волны (нм)")
        plt.ylabel("Интенсивность (Вт/м²·нм)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "spectrum_comparison.png"))
        plt.close()

    def plot_error_by_wavelength(self, error_df):
        plt.figure(figsize=(14, 5))
        plt.plot(error_df["Wavelength"], error_df["Error (%)"], label="Ошибка (%)")
        plt.axhline(y=15, color='r', linestyle='--', label="Допустимая ошибка (15%)")
        plt.title("Ошибка по длине волны")
        plt.xlabel("Длина волны (нм)")
        plt.ylabel("Ошибка (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "spectrum_error_by_wavelength.png"))
        plt.close()