# process_data.py
# Usage: python process_data.py
import os
import numpy as np
import pandas as pd

# ---- USER EDITABLE ----
INPUT_CSV = os.path.join('data', 'bmi.csv')  # Put your original dataset here
OUTPUT_CSV = os.path.join('summative', 'linear_regression', 'processed_bmi.csv')
RANDOM_SEED = 42
DIST_MIN = 1.0
DIST_MAX = 10.0
TERRAIN_VALUES = [1.0, 1.3, 1.6]
TERRAIN_PROBS = [0.55, 0.35, 0.10]
ROLLING_CONST = 0.35  # Wh per kg per km baseline
# ------------------------

def main():
    assert os.path.exists(INPUT_CSV), f"Input file not found at {INPUT_CSV}. Place your raw bmi.csv there."
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    # Basic sanity check
    expected_cols = {'Age','Height','Weight','Bmi','BmiClass'}
    missing = expected_cols - set(df.columns)
    if missing:
        print("Warning: expected columns missing:", missing)
        # proceed but Weight must exist
    if 'Weight' not in df.columns:
        raise KeyError("Weight column required to compute battery_Wh")

    np.random.seed(RANDOM_SEED)
    n = len(df)

    df_proc = df.copy()
    df_proc['daily_distance_km'] = np.round(np.random.uniform(DIST_MIN, DIST_MAX, size=n), 3)
    df_proc['terrain_factor'] = np.random.choice(TERRAIN_VALUES, size=n, p=TERRAIN_PROBS)
    df_proc['battery_Wh'] = df_proc['Weight'] * ROLLING_CONST * df_proc['terrain_factor'] * df_proc['daily_distance_km']

    df_proc.to_csv(OUTPUT_CSV, index=False)
    print(f"Processed dataset saved to: {OUTPUT_CSV}")
    print("battery_Wh stats:")
    print(df_proc['battery_Wh'].describe().round(3))

if __name__ == "__main__":
    main()
