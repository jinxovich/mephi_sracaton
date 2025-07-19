import numpy as np
import pandas as pd
import os

# Путь к папке с источниками
OUTPUT_DIR = "config/sources"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Диапазон длин волн
WAVELENGTH_RANGE = np.arange(380, 1101)

# Улучшенные и дополненные типы источников
LED_TYPES = [
    # Название, пиковая длина волны, ширина пика (sigma), амплитуда
    {"name": "Broad_White", "peak": 550, "sigma": 100, "amp": 1.0},  # База
    {"name": "Cool_White", "peak": 460, "sigma": 20, "amp": 0.9},
    {"name": "Warm_White", "peak": 580, "sigma": 30, "amp": 0.85},
    {"name": "Red", "peak": 660, "sigma": 15, "amp": 1.0},
    {"name": "Far_Red", "peak": 730, "sigma": 15, "amp": 0.7},
    {"name": "Deep_Red", "peak": 620, "sigma": 10, "amp": 0.9},
    {"name": "Green", "peak": 520, "sigma": 10, "amp": 0.95},
    {"name": "Green_550", "peak": 550, "sigma": 10, "amp": 0.95},
    {"name": "Blue", "peak": 450, "sigma": 10, "amp": 0.9},
    {"name": "Deep_Blue", "peak": 410, "sigma": 10, "amp": 0.8},
    {"name": "Amber", "peak": 590, "sigma": 10, "amp": 0.9},
    {"name": "Yellow", "peak": 570, "sigma": 10, "amp": 0.9},
    {"name": "Cyan", "peak": 490, "sigma": 10, "amp": 0.9},
    {"name": "Teal", "peak": 495, "sigma": 10, "amp": 0.8},
    {"name": "IR_850", "peak": 850, "sigma": 20, "amp": 0.7},
    {"name": "IR_940", "peak": 940, "sigma": 20, "amp": 0.6},
    {"name": "UV_380", "peak": 380, "sigma": 10, "amp": 0.5},
    {"name": "UV_350", "peak": 350, "sigma": 10, "amp": 0.6},
    {"name": "NIR_1050", "peak": 1050, "sigma": 20, "amp": 0.6},
    {"name": "NIR_1200", "peak": 1200, "sigma": 20, "amp": 0.5},
    {"name": "Lime", "peak": 510, "sigma": 10, "amp": 0.85},
    {"name": "Pink", "peak": 540, "sigma": 20, "amp": 0.8},
    {"name": "White_4000K", "peak": 550, "sigma": 60, "amp": 0.95},
    {"name": "White_5000K", "peak": 550, "sigma": 60, "amp": 0.95},
    {"name": "White_6500K", "peak": 550, "sigma": 60, "amp": 0.95},
    {"name": "White_7500K", "peak": 550, "sigma": 60, "amp": 0.95},
    {"name": "Orange", "peak": 590, "sigma": 10, "amp": 0.9},
    {"name": "Violet", "peak": 410, "sigma": 10, "amp": 0.8},
    {"name": "Purple", "peak": 430, "sigma": 10, "amp": 0.8},
    {"name": "Deep_Orange", "peak": 600, "sigma": 10, "amp": 0.9},
    {"name": "Laser_Blue", "peak": 450, "sigma": 2, "amp": 1.0},  # Очень узкий пик
    {"name": "Laser_Red", "peak": 660, "sigma": 2, "amp": 1.0},
    {"name": "Laser_Green", "peak": 520, "sigma": 2, "amp": 1.0},
    {"name": "Laser_IR", "peak": 940, "sigma": 2, "amp": 0.7},
    {"name": "Laser_UV", "peak": 365, "sigma": 2, "amp": 0.6},
    {"name": "Wide_Green", "peak": 520, "sigma": 30, "amp": 0.8},
    {"name": "Wide_Red", "peak": 660, "sigma": 30, "amp": 0.8},
    {"name": "Wide_Blue", "peak": 450, "sigma": 30, "amp": 0.8},
    {"name": "Wide_IR", "peak": 940, "sigma": 50, "amp": 0.7},
    {"name": "Wide_UV", "peak": 380, "sigma": 20, "amp": 0.6},
    {"name": "Flat_Low", "peak": 550, "sigma": 200, "amp": 0.7},  # Широкий и плоский
    {"name": "Flat_High", "peak": 550, "sigma": 200, "amp": 1.0},
    {"name": "Tri_450", "peak": 450, "sigma": 15, "amp": 1.0},
    {"name": "Tri_550", "peak": 550, "sigma": 15, "amp": 1.0},
    {"name": "Tri_650", "peak": 650, "sigma": 15, "amp": 1.0},
    {"name": "Tri_850", "peak": 850, "sigma": 15, "amp": 0.7},
    {"name": "Tri_1000", "peak": 1000, "sigma": 15, "amp": 0.6},
    {"name": "Tri_1200", "peak": 1200, "sigma": 15, "amp": 0.5},
    {"name": "Fant1", "peak": 820, "sigma": 5, "amp": 0.2},
    {"name": "Fant2", "peak": 960, "sigma": 5, "amp": 0.2},
    {"name": "Fant3", "peak": 1100, "sigma": 5, "amp": 0.2},
    {"name": "Fant4", "peak": 960, "sigma": 10, "amp": 0.3},
    {"name": "Fant5", "peak": 760, "sigma": 2, "amp": 0.2},
    {"name": "Fant6", "peak": 720, "sigma": 1, "amp": 0.1},
    {"name": "Fant7", "peak": 800, "sigma": 1, "amp": 0.1},
    {"name": "Fant8", "peak": 880, "sigma": 1, "amp": 0.1},
    {"name": "Fant9", "peak": 950, "sigma": 1, "amp": 0.1},
    {"name": "Fant10", "peak": 920, "sigma": 3, "amp": 0.1},
    {"name": "Fant11", "peak": 777, "sigma": 2, "amp": 0.05},
    {"name": "Fant12", "peak": 950, "sigma": 1, "amp": 1.5},
    {"name": "Fant13", "peak": 850, "sigma": 1, "amp": 2}


]

# Добавляем новые источники
LED_TYPES.extend([
    {"name": "NIR_980", "peak": 980, "sigma": 20, "amp": 0.6},  # Дополнительный ИК-источник
    {"name": "NIR_1100", "peak": 1100, "sigma": 20, "amp": 0.5},  # Дополнительный ИК-источник
    {"name": "Deep_Red_700", "peak": 700, "sigma": 15, "amp": 0.8},  # Дополнительный красный источник
    {"name": "UV_365", "peak": 365, "sigma": 10, "amp": 0.7},  # УФ-источник
    {"name": "Wide_IR", "peak": 940, "sigma": 50, "amp": 0.7},  # Широкий ИК-спектр
    {"name": "Wide_UV", "peak": 380, "sigma": 20, "amp": 0.6},  # Широкий УФ-спектр
])

def gaussian(x, mu, sigma, amp=1.0):
    """Гауссово распределение для моделирования спектра"""
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def generate_led_spectrum(peak, sigma, amp, name):
    """Генерация синтетического спектра с шумом и нормализацией"""
    intensity = gaussian(WAVELENGTH_RANGE, peak, sigma, amp)
    intensity = np.clip(intensity, 0, None)

    # Добавляем шум для разнообразия
    noise = np.random.normal(0, intensity.max() * 0.02, size=intensity.shape)
    intensity += noise
    intensity = np.clip(intensity, 0, None)

    df = pd.DataFrame({
        "Wavelength": WAVELENGTH_RANGE,
        "Intensity": intensity
    })

    # Проверяем, что спектр покрывает весь диапазон
    if not np.any(df["Intensity"] > 0):
        print(f"⚠️ Спектр {name} не содержит значений выше 0")
        return

    filename = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(filename, index=False)
    print(f"Сохранён файл: {filename}")

def main():
    print(f"Генерация спектров для {len(LED_TYPES)} источников...")
    for led in LED_TYPES:
        generate_led_spectrum(
            peak=led["peak"],
            sigma=led["sigma"],
            amp=led["amp"],
            name=led["name"]
        )
    print("✅ Генерация завершена.")

if __name__ == "__main__":
    main()