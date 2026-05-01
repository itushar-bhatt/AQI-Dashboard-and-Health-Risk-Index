from src.processing import process_air_quality_data
from src.database import save_data, load_data

df = process_air_quality_data('data/aqi2.csv')

save_data(df)

print(load_data().head())