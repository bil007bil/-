import os
import pandas as pd
import glob

# Путь к вашей папке с CSV
BASE_PATH = r"E:\Рабочий стол\ГПНИ_БРФФИ\Layers"
OUTPUT_FILE = "all_layers_database.pkl"


def collect_data():
    all_files = glob.glob(os.path.join(BASE_PATH, "*.csv"))
    if not all_files:
        print("Файлы не найдены!")
        return

    combined_data = []

    for file in all_files:
        layer_name = os.path.basename(file).split('.')[0]
        try:
            # Читаем CSV с учетом ваших особенностей (B, C, D, H)
            df = pd.read_csv(file, sep=None, engine='python', decimal=',',
                             usecols=[1, 2, 3, 7],
                             names=['lat', 'lon', 'ustye', 'abs_z'],
                             header=0)

            df = df.dropna(subset=['lat', 'lon', 'abs_z'])
            df['layer'] = layer_name  # Добавляем колонку с названием слоя
            combined_data.append(df)
            print(f"Считан слой: {layer_name}")
        except Exception as e:
            print(f"Ошибка в файле {layer_name}: {e}")

    if combined_data:
        full_df = pd.concat(combined_data, ignore_index=True)
        # Сохраняем в бинарный формат для мгновенной загрузки
        full_df.to_pickle(OUTPUT_FILE)
        print(f"\nГотово! База сохранена в {OUTPUT_FILE}")
        print(f"Всего точек в базе: {len(full_df)}")


if __name__ == "__main__":
    collect_data()