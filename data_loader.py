import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir="data", config_dir="config"):
        self.data_dir = data_dir
        self.config_dir = config_dir

    def load_target_spectrum(self):
        file_path = os.path.join(self.data_dir, "am1_5g_spectrum.csv")
        df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1, nrows=1)

        if "Wavelength (nm)" in df.columns:
            df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1)
            df.columns = ["Wavelength", "Spectral_irradiance"]
        else:
            df = pd.read_csv(file_path, sep=None, engine='python', nrows=1)
            if df.columns[0].lower().startswith("wavelength"):
                df = pd.read_csv(file_path, sep=None, engine='python', skiprows=1)
                df.columns = ["Wavelength", "Spectral_irradiance"]
            else:
                df = pd.read_csv(file_path, sep=None, engine='python', header=None)
                df.columns = ["Wavelength", "Spectral_irradiance"]

        df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
        df["Spectral_irradiance"] = pd.to_numeric(df["Spectral_irradiance"], errors="coerce")
        return df.dropna().reset_index(drop=True)

    def load_source_spectra(self):
        sources_path = os.path.join(self.config_dir, "sources")
        source_files = [f for f in os.listdir(sources_path) if f.endswith(".csv")]

        sources = []
        for file in source_files:
            file_path = os.path.join(sources_path, file)
            df = pd.read_csv(file_path, sep=None, engine='python', nrows=1)

            if "Wavelength" in df.columns and "Intensity" in df.columns:
                df = pd.read_csv(file_path, sep=None, engine='python')
            else:
                df = pd.read_csv(file_path, sep=None, engine='python', header=None)
                df.columns = ["Wavelength", "Intensity"]

            df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
            df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
            df = df.dropna().reset_index(drop=True)

            if df["Intensity"].max() < 0.01:
                print(f"Источник {file} игнорируется — слишком слабый")
                continue

            sources.append({
                "name": os.path.splitext(file)[0],
                "spectrum": df[["Wavelength", "Intensity"]]
            })

        return sources