This project is for the development of a model to predict the power output of a combined cycle power plant using information from the sensors.

## Data
The data for this project is taken from the UCI machine learning repository and contains 9568 data points with information for 5 variables. (4 features and 1 output). 
Features consist of hourly average ambient variables
- Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in the range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW
The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.

Source: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant


## Model
The model is developed using an Artificial Neural Network framework. This framework is chosen because the variables are known to have a complex relation with Power. A linear model would not be able to capture the non-linear effects without targeted feature engineering. A Random Forest Regressor was attempted but did not reach the required performance level. Since this was a large dataset, an ANN approach would be a good approach to fit the dataset and let the model fit the underlying parameters. 

## Results
The metrics to analyze the data was done using r2. It as found that the r2 of the result was ~91%. The out-of-sample and in-sample MSE were close, indicating a low variance. 