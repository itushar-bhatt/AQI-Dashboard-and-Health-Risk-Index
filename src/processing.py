import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder


def process_air_quality_data(file_path: str) -> pd.DataFrame:

    # 1. Load the data
    df = pd.read_csv(file_path)

    cols = [
        'city', 'state', 'latitude', 'longitude', 'datetime',
        'month', 'day_name', 'season', 'time_of_day', 'us_aqi', 'festival_period'
    ]

    df = df[cols].rename(
        columns={'us_aqi': 'aqi_value', 'city': 'area', 'datetime': 'date'}
    )


    # 2. Date conversion
    df['date'] = pd.to_datetime(df['date'])
    df['was_missing'] = df['aqi_value'].isna()

    df = df.sort_values(by=['area', 'date']).reset_index(drop=True)

    #
    festival_dates = [
        "2022-10-25",
        "2023-11-12",
        "2024-10-20"
    ]
    festival_dates = pd.to_datetime(festival_dates)

    df['festivals_period'] = 0

    for d in festival_dates:
        df.loc[
            df['date'].between(d - pd.Timedelta(days=1),
                            d + pd.Timedelta(days=5)),
            'festivals_period'
        ] = 1
        
        
    # 3. Spike Detection
    def analyze_city(city_df):

        city_df = city_df.copy()
        city_df['aqi_jump'] = city_df['aqi_value'].diff().abs()

        if len(city_df) < 20:
            city_df['expected_aqi'] = city_df['aqi_value'].fillna(
                city_df['aqi_value'].mean()
            )
            return city_df

        city_df['month_year'] = city_df['date'].dt.to_period('M').astype(str)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        month_encoded = encoder.fit_transform(city_df[['month_year']])

        time_idx = np.arange(len(city_df)).reshape(-1, 1)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        time_poly = poly.fit_transform(time_idx)

        X = np.hstack([time_poly, month_encoded])

        y = city_df['aqi_value'].fillna(city_df['aqi_value'].mean()).values

        model = LinearRegression()
        model.fit(X, y)

        city_df['expected_aqi'] = model.predict(X)

        return city_df

    print("Starting Modeling (OHE + Polynomial + Spikes)")

    # Save area column (BACKUP)
    area_backup = df['area'].copy()

    # Run the ML
    df = df.groupby('area', group_keys=False).apply(analyze_city)

    # Reset index
    df = df.reset_index(drop=True)

    # Safety Check
    if 'area' not in df.columns:
        df['area'] = area_backup.values
        print("DEBUG: Area column was restored from backup")

    print(f"DEBUG: Columns after ML: {list(df.columns)}")

    # 4. Anomaly Detection
    df['residual'] = (df['aqi_value'] - df['expected_aqi']).abs()

    city_std = df.groupby('area')['residual'].transform('std')

    df['is_anomaly'] = np.where(
    (
        ((df['residual'] > 3 * city_std) | (df['aqi_jump'] > 150))
        & (df['festivals_period'] == 0)
    ),
    "Yes",
    "No"
)

    # 5. Data Correction
    df['data_origin'] = np.where(
        (df['is_anomaly'] == "Yes") | (df['was_missing']),
        "Guessed",
        "Actual"
    )

    df['aqi_value'] = np.where(
        (df['is_anomaly'] == "Yes") | (df['was_missing']),
        df['expected_aqi'],
        df['aqi_value']
    )

    df['aqi_value'] = df['aqi_value'].clip(lower=0).round().astype(int)

    # 6. Status
    h_condis = [
        (df['aqi_value'] <= 50),
        (df['aqi_value'] <= 100),
        (df['aqi_value'] <= 200),
        (df['aqi_value'] <= 300)
    ]

    h_labels = [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor / Unhealthy"
    ]

    df['health_risk_category'] = np.select(
        h_condis,
        h_labels,
        default="Severe / Hazardous"
    )

    status_condis = [
        (df['is_anomaly'] == "Yes"),
        (df['was_missing'] == True)
    ]

    status_labels = [
        "Anomaly Detected",
        "Missing"
    ]

    df['sensor_status'] = np.select(
        status_condis,
        status_labels,
        default="Operational"
    )

    df = df.drop(columns=[
        'expected_aqi',
        'residual',
        'aqi_jump',
        'was_missing'
    ])

    #check duplicates
    dup = df[df.duplicated()]
    print(dup)
    print(df.columns)

    print(f"{len(df)} rows processed")

    return df