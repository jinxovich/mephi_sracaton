import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tqdm import tqdm

class SpectrumOptimizer:
    def __init__(self, target_spectrum, sources, wavelength_range=(380, 1100), max_sources=10, tolerance=0.15):
        """
        Основной класс для подбора минимального набора источников света,
        чей комбинированный спектр воспроизводит эталонный спектр AM1.5G
        """
        # Фильтруем эталонный спектр по диапазону
        self.target_spectrum = target_spectrum[
            (target_spectrum["Wavelength"] >= wavelength_range[0]) &
            (target_spectrum["Wavelength"] <= wavelength_range[1])
        ].reset_index(drop=True)

        self.sources = sources
        self.max_sources = max_sources
        self.tolerance = tolerance
        self.target_wavelengths = self.target_spectrum["Wavelength"].values
        self.target_values = self.target_spectrum["Spectral_irradiance"].values
        
        # Нормализуем эталонный спектр для согласованности масштабов
        self.target_max = self.target_values.max()
        self.target_values_normalized = self.target_values / self.target_max
        
        print(f"📊 Эталонный спектр: мин={self.target_values.min():.4f}, макс={self.target_values.max():.4f}")
        print(f"📊 После нормализации: мин={self.target_values_normalized.min():.4f}, макс={self.target_values_normalized.max():.4f}")

        # Нормализуем все источники
        self.normalized_sources = self._normalize_sources()

    def _normalize_sources(self):
        """Нормализуем интенсивность каждого источника по максимальному значению"""
        normalized = []
        for source in self.sources:
            spectrum = source["spectrum"].copy()
            max_intensity = spectrum["Intensity"].max()
            if max_intensity > 0:
                spectrum["Intensity"] /= max_intensity
            normalized.append({"name": source["name"], "spectrum": spectrum})
        return normalized

    def _interpolate_spectrum(self, spectrum):
        """Интерполирует спектр на эталонную шкалу длин волн"""
        return np.interp(self.target_wavelengths, spectrum["Wavelength"], spectrum["Intensity"])

    def _calculate_error(self, combined_values):
        """Вычисляет среднюю и максимальную относительную ошибку с защитой от деления на ноль"""
        # Используем нормализованные значения эталона
        target_safe = np.where(np.abs(self.target_values_normalized) < 1e-10, 1e-10, self.target_values_normalized)
        
        # Вычисляем относительную ошибку
        relative_error = np.abs((combined_values - self.target_values_normalized) / target_safe)
        
        # Дополнительная проверка на аномально большие ошибки
        relative_error = np.where(relative_error > 100, 100, relative_error)
        
        mean_error = np.mean(relative_error)
        max_error = np.max(relative_error)
        
        return mean_error, max_error

    def _calculate_local_error(self, combined_values):
        """Возвращает DataFrame с ошибкой по каждой длине волны"""
        target_safe = np.where(np.abs(self.target_values_normalized) < 1e-10, 1e-10, self.target_values_normalized)
        error = np.abs((combined_values - self.target_values_normalized) / target_safe) * 100
        
        # Ограничиваем максимальную ошибку для визуализации
        error = np.where(error > 1000, 1000, error)
        
        return pd.DataFrame({
            "Wavelength": self.target_wavelengths,
            "Error (%)": error
        })

    def _calculate_correlation(self, source):
        """Возвращает корреляцию между источником и эталоном"""
        interp = self._interpolate_spectrum(source["spectrum"])
        
        # Проверяем на NaN и константные значения
        if np.all(interp == 0) or np.all(self.target_values_normalized == self.target_values_normalized[0]):
            return 0
        
        try:
            corr_matrix = np.corrcoef(interp, self.target_values_normalized)
            correlation = corr_matrix[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0

    def _add_source_to_combo(self, current_combo):
        """Добавляет один новый источник к текущей комбинации"""
        best_source = None
        best_coefficients = None
        best_error = float('inf')
        best_spectrum = None

        for source in tqdm(self.normalized_sources, desc="Поиск лучшего источника"):
            if source in current_combo:
                continue

            A = np.column_stack([
                self._interpolate_spectrum(s["spectrum"]) for s in current_combo + [source]
            ])

            try:
                coefficients, residual = nnls(A, self.target_values_normalized)
                combined_values = A @ coefficients
                mean_error, max_error = self._calculate_error(combined_values)

                if max_error < best_error:
                    best_error = max_error
                    best_source = source
                    best_coefficients = coefficients
                    best_spectrum = combined_values

            except Exception as e:
                print(f"⚠️ Ошибка при обработке источника {source['name']}: {e}")
                continue

        return best_source, best_coefficients, best_error, best_spectrum

    def greedy_optimize(self):
        """
        Жадный алгоритм с пересчётом коэффициентов на каждом шаге
        """
        print("\nНачинаем улучшенный жадный поиск...")

        # Сортируем источники по корреляции с эталоном
        print("🔍 Ранжируем источники по корреляции...")
        correlations = []
        for source in self.normalized_sources:
            corr = self._calculate_correlation(source)
            correlations.append((source, corr))
            
        # Выводим топ-10 источников по корреляции
        correlations.sort(key=lambda x: x[1], reverse=True)
        print("📈 Топ-10 источников по корреляции с эталоном:")
        for i, (source, corr) in enumerate(correlations[:10]):
            print(f"  {i+1}. {source['name']}: {corr:.4f}")
        
        ranked_sources = [item[0] for item in correlations]

        current_combo = []
        best_error = float('inf')
        best_coefficients = None
        best_spectrum = None

        # На каждом шаге добавляем один источник, который максимально уменьшает ошибку
        for step in range(1, self.max_sources + 1):
            best_step_source = None
            best_step_coefficients = None
            best_step_error = float('inf')
            best_step_spectrum = None

            print(f"\nШаг {step}: добавляем {step}-й источник")

            for source in tqdm(ranked_sources, desc=f"Проверка источников (шаг {step})"):
                if source in current_combo:
                    continue

                A = np.column_stack([
                    self._interpolate_spectrum(s["spectrum"]) for s in current_combo + [source]
                ])

                try:
                    coefficients, residual = nnls(A, self.target_values_normalized)
                    combined_values = A @ coefficients
                    mean_error, max_error = self._calculate_error(combined_values)

                    if max_error < best_step_error:
                        best_step_error = max_error
                        best_step_source = source
                        best_step_coefficients = coefficients
                        best_step_spectrum = combined_values

                except Exception as e:
                    continue

            if best_step_source is None:
                print("⚠️ Не найдено подходящих источников")
                break

            current_combo.append(best_step_source)
            best_coefficients = best_step_coefficients
            best_error = best_step_error
            best_spectrum = best_step_spectrum

            print(f"Добавлен: {best_step_source['name']}, ошибка: {best_step_error:.4f}")

            if best_step_error <= self.tolerance:
                print("✅ Допустимая ошибка достигнута!")
                break

        if not current_combo:
            print("⚠️ Не удалось найти комбинацию с допустимой ошибкой")
            return {
                "sources": [],
                "coefficients": np.array([]),
                "mean_error": float('inf'),
                "max_error": float('inf'),
                "combined_spectrum": pd.DataFrame({
                    "Wavelength": self.target_wavelengths,
                    "Spectral_irradiance": np.zeros_like(self.target_values)
                }),
                "error_by_wavelength": self._calculate_local_error(np.zeros_like(self.target_values_normalized))
            }

        # Восстанавливаем исходный масштаб для результата
        best_spectrum_rescaled = best_spectrum * self.target_max

        combined_spectrum = pd.DataFrame({
            "Wavelength": self.target_wavelengths,
            "Spectral_irradiance": best_spectrum_rescaled
        })

        error_by_wavelength = self._calculate_local_error(best_spectrum)

        # Оставляем только те коэффициенты, которые соответствуют текущей комбинации
        final_coefficients = best_coefficients[:len(current_combo)]
        
        print(f"\n📊 Финальная статистика:")
        print(f"   Средняя ошибка: {self._calculate_error(best_spectrum)[0]:.4f}")
        print(f"   Максимальная ошибка: {best_error:.4f}")
        print(f"   Количество источников: {len(current_combo)}")

        return {
            "sources": current_combo,
            "coefficients": final_coefficients,
            "mean_error": self._calculate_error(best_spectrum)[0],
            "max_error": best_error,
            "combined_spectrum": combined_spectrum,
            "error_by_wavelength": error_by_wavelength
        }