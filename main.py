import pandas as pd
import numpy as np

from data_loader import DataLoader
from optimizer import SpectrumOptimizer
from visualizer import SpectrumVisualizer

def main():
    print("Загрузка данных...")
    data_loader = DataLoader()
    target_spectrum = data_loader.load_target_spectrum()
    sources = data_loader.load_source_spectra()

    print("Начало оптимизации...")
    optimizer = SpectrumOptimizer(
        target_spectrum=target_spectrum,
        sources=sources,
        max_sources=200,  # <-- Максимальное количество источников
        tolerance=0.15
    )

    result = optimizer.greedy_optimize()

    if not result["sources"] or len(result["sources"]) == 0:
        print("⚠️ Не удалось найти комбинацию с допустимой ошибкой.")
        return

    print(f"\n✅ Оптимальная комбинация найдена!")
    print(f"Количество источников: {len(result['sources'])}")
    print(f"Максимальная ошибка: {result['max_error']:.4f}")
    print("\nИсточники и коэффициенты:")

    for source, coeff in zip(result["sources"], result["coefficients"]):
        print(f"{source['name']}: {coeff:.4f}")

    print("Сохранение графиков...")
    visualizer = SpectrumVisualizer()
    visualizer.plot_spectra(target_spectrum, result["combined_spectrum"])
    if "error_by_wavelength" in result and not result["error_by_wavelength"].empty:
        visualizer.plot_error_by_wavelength(result["error_by_wavelength"])

    print("✅ Анализ завершён.")

if __name__ == "__main__":
    main()