# linear_regression_model

Mission: Use technology to provide personalized powered mobility (automatic wheelchairs) by predicting the battery energy (Wh) required for a user from body measurements and daily usage profile.  
Dataset: `processed_bmi.csv` (based on an anthropometry dataset containing Age, Height, Weight, BMI and engineered usage features).  
Processing: `daily_distance_km` and `terrain_factor` were added (seeded sampling) and `battery_Wh` was computed using an explained physics-based formula.  
Deliverables: multivariate.ipynb, FastAPI endpoint (Swagger UI), Flutter one-page app, and ≤5-minute demo video.


## Structure

```
linear_regression_model/
└── summative/
    ├── linear_regression/
    │   ├── multivariate.ipynb            # notebook (EDA, training, plots)
    │   ├── processed_bmi.csv             # processed dataset (Age,Height,Weight,Bmi,BmiClass,daily_distance_km,terrain_factor,battery_Wh)
    │   └── model_metrics.csv
    ├── API/
    │   ├── prediction.py                 # FastAPI app
    │   ├── requirements.txt
    │   ├── best_model.pkl                # saved best model (RandomForest)
    │   └── scaler.pkl                    # saved StandardScaler
    └── FlutterApp/
        ├── lib/
        │   └── main.dart                
        └── README.md                     #Flutter Readme
README.md
```
