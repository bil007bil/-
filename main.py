import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

DB_FILE = "all_layers_database.pkl"


class FastGeoPredictor:
    def __init__(self):
        if not os.path.exists(DB_FILE):
            print(f"Файл {DB_FILE} не найден! Сначала запустите collector.py")
            exit()

        print("Загрузка базы данных...")
        self.df = pd.read_pickle(DB_FILE)
        self.models = {}
        self.layer_names = self.df['layer'].unique()

        # Модель поверхности (устья) обучается по всем уникальным точкам координат
        print("Инициализация модели рельефа...")
        surface_df = self.df[['lat', 'lon', 'ustye']].drop_duplicates(subset=['lat', 'lon'])
        self.surface_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.surface_model.fit(surface_df[['lat', 'lon']], surface_df['ustye'])
        self.surface_data = surface_df

    def predict_for_point(self, lat, lon):
        results = []

        # 1. Проверяем устье (факт или прогноз)
        exact_surf = self.surface_data[(np.isclose(self.surface_data['lat'], lat, atol=0.1)) &
                                       (np.isclose(self.surface_data['lon'], lon, atol=0.1))]

        z_surf = exact_surf.iloc[0]['ustye'] if not exact_surf.empty else self.surface_model.predict([[lat, lon]])[0]
        results.append({'Слой': '--- ПОВЕРХНОСТЬ (Устье) ---', 'Z': round(z_surf, 2),
                        'Тип': 'ФАКТ' if not exact_surf.empty else 'ПРОГНОЗ'})

        # 2. Проходим по слоям
        print("Расчет по слоям...")
        for layer in self.layer_names:
            layer_df = self.df[self.df['layer'] == layer]

            # Проверяем, есть ли эта точка в конкретном слое
            exact_layer = layer_df[(np.isclose(layer_df['lat'], lat, atol=0.1)) &
                                   (np.isclose(layer_df['lon'], lon, atol=0.1))]

            if not exact_layer.empty:
                z_val = exact_layer.iloc[0]['abs_z']
                results.append({'Слой': layer, 'Z': round(z_val, 2), 'Тип': 'ФАКТ'})
            else:
                # Обучаем модель "на лету" только для нужных слоев или можно обучить заранее
                # Для скорости здесь лучше использовать упрощенный прогноз или заранее обученные модели
                # Но так как слоев 460, обучим только те, где есть данные
                model = RandomForestRegressor(n_estimators=20, random_state=42)
                model.fit(layer_df[['lat', 'lon']], layer_df['abs_z'])
                z_val = model.predict([[lat, lon]])[0]
                results.append({'Слой': layer, 'Z': round(z_val, 2), 'Тип': 'ПРОГНОЗ'})

        return pd.DataFrame(results).sort_values(by='Z', ascending=False)


# --- ЗАПУСК ---
if __name__ == "__main__":
    predictor = FastGeoPredictor()

    try:
        lat_in = float(input("\nВведите Широту (B): ").replace(',', '.'))
        lon_in = float(input("Введите Долготу (C): ").replace(',', '.'))

        final_res = predictor.predict_for_point(lat_in, lon_in)

        print("\n" + "=" * 50)
        print(final_res.to_string(index=False))
        print("=" * 50)

        final_res.to_csv("result_point.csv", sep=';', index=False, encoding='utf-8-sig')
        print(f"Результат сохранен в 'result_point.csv'")

    except Exception as e:
        print(f"Ошибка: {e}")