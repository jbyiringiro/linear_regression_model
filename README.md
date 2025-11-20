# linear_regression_model


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
    │   ├── best_model.pkl  
    │   └── scaler.pkl                    # saved StandardScaler
    └── FlutterApp/
        ├──lrmodelapp/                    #Flutter App
            ├── lib/
            │   └── main.dart                
            └── README.md                     #Flutter Readme
README.md
```
