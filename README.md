# NASA Airport Throughput Prediction Challenge

This project aims to predict airport throughput using a combination of advanced machine learning techniques, including LSTM models and XGBoost regressors. The project is part of the NASA Airport Throughput Prediction Challenge, addressing the complexities of air traffic and airport operations. The approach leverages time-series forecasting and feature engineering to achieve accurate predictions.


## Features

- **LSTM Neural Networks**: A deep learning approach to capture sequential dependencies in airport data.

- **XGBoost Regressor**: A gradient-boosted decision tree model to complement the deep learning approach and provide robust predictions.

- **Custom TensorFlow Layers**: Includes a custom input-weighting layer to dynamically adjust feature importance during training.

- **Threaded Data Processing**: Optimized for concurrent data preparation and prediction using `ThreadPoolExecutor`.

- **Feature Engineering**: Data normalization, time-based feature extraction, and one-hot encoding for enhanced model input.


## Installation

1. Clone the repository:

       git clone https://github.com/SamuelTWu/nasaATPchallenge.git
       cd nasaATPchallenge

2. Install the required Python packages:

       pip install -r requirements.txt

3. Ensure your environment supports TensorFlow and XGBoost.


## Data Preparation

The dataset consists of multiple features related to airport operations, such as weather, runways, and air traffic data. Follow these steps to preprocess the data:

1. Use `get_train_data` to load and preprocess the training dataset from various folders (`FUSER_train`).

2. Align weather data with target data using `AIRPORT_DATE_TIME` and index.

3. Apply one-hot encoding and feature scaling using `MinMaxScaler`.


## Models

### LSTM Model

The LSTM model is designed to handle multivariate time-series data, capturing temporal dependencies effectively. It includes:

- A custom `InputWeightingLayer` to dynamically adjust feature weights.

- Four LSTM layers with `Dropout` for regularization.

- A `Dense` output layer with a linear activation function.

**Key Parameters**:

- Input sequence length (`n_steps_in`): 1440

- Output sequence length (`n_steps_out`): 1

- Optimizer: Adam with a learning rate of 0.0001

**Training**:

- Model is trained using Mean Squared Error (MSE) as the loss function.


### XGBoost Regressor

The XGBoost model serves as a robust machine learning alternative to the LSTM model. It handles tabular data and captures nonlinear relationships effectively.

**Key Features**:

- Gradient boosting algorithm for accurate predictions.

- Tunable hyperparameters for optimization.

**Training**:

- Train the model using preprocessed data:

      xgb_model = XGBRegressor(**xgb_params)
      xgb_model = xgb_model.fit(X_train, Y_train)
      xgb_model.save_model("models/xgb_model_n30000_d15_lr1.json")

**Evaluation**:

- Root Mean Squared Error (RMSE)

- R-squared Score

<!---->

    predictions = xgb_model.predict(X_test)
    rmse = root_mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    print("Root Mean Squared Error:", rmse)
    print("R-squared Score:", r2)


## Threaded Data Processing

To handle the large dataset efficiently, the project uses multithreading for data processing and prediction. The `ThreadPoolExecutor` manages concurrent threads, ensuring seamless execution.


## Submission

The project includes a utility to format predictions and save them as a CSV file for submission:

    submission_format_df = pd.read_csv("data/submission_format.csv")
    post_process(submission_format_df, predict_df).to_csv("data/submission_w64_l3_128_s720_b1024.csv", index=False)


## Results

### LSTM
  We experimented with different components of the LSTM to see which hyperparameters produced the best predictions. We experimented with LSTM layers from 2-4, timesteps in from 100-720, LSTM layer size from 64 to 128, initial weight size from 1 to 64, and batch size from 256 to 1024. Our best performing model had an initial weight of 64, 3 LSTM layers of size 64, in steps of 420, and batch size of 1024, with a RMSE of 5.299 and a competition score of 0.589. 

### XGBoost Regressor
  We also experimented with an XGBoost Regressor. We chose this model because it trains very quickly and can handle very large amounts of data. We experimented with XGBoost models of differing n_estimators and max_depths, with the highest performing model scoring a RMSE of 3.69 and a competition score of .692. 



## Acknowledgments

This project is part of the NASA Airport Throughput Prediction Challenge and leverages data provided by NASA. Special thanks to the organizers for their support and resources.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

For any questions or contributions, please reach out to Samuel Wu at samwu@bu.edu
