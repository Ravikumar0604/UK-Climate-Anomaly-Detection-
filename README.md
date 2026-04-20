# Multivariate Climate Anomaly Detection in the UK Using Machine Learning and Deep Learning



## Project Overview
This project investigates climate anomaly detection across major cities in the United Kingdom using multivariate time-series climate data. The study compares a classical unsupervised machine learning model, **Isolation Forest**, with a deep learning temporal model, **LSTM Autoencoder**, to determine which approach provides more reliable anomaly detection.

The dataset was collected from the **NASA POWER** platform for five UK cities from **2016 to 2022**. The modelling pipeline includes data validation, preprocessing, exploratory data analysis, feature engineering, sequence generation, baseline model development, model tuning, evaluation, and final model comparison.

The final results showed that the **Tuned LSTM Autoencoder** achieved the best performance, producing the highest anomaly separation and better generalisation on unseen data.

---

## Research Aim
To evaluate and compare the effectiveness of classical and deep learning-based anomaly detection models for multivariate climate time-series data across UK cities.

---

## Research Questions
1. Which model provides the most reliable climate anomaly detection across UK cities: **Isolation Forest** or **LSTM-based deep learning models**?
2. To what extent does modelling temporal dependencies with **LSTM** improve anomaly detection compared to non-temporal methods such as **Isolation Forest**?

---

## Dataset
- **Source:** NASA POWER
- **Cities:** Birmingham, Bristol, Leeds, London, Manchester
- **Time Period:** 2016–2022
- **Data Type:** Daily multivariate climate observations

### Core Variables
- `T2M` — Surface air temperature
- `T2M_MAX` — Maximum daily temperature
- `T2M_MIN` — Minimum daily temperature
- `PRECTOTCORR` — Precipitation
- `RH2M` — Relative humidity
- `WS2M` — Wind speed

### Validation
The NASA POWER dataset was externally validated using **UK Met Office** observations for London stations:
- Heathrow
- Kew Gardens
- Northolt

Validation confirmed strong alignment between satellite-derived and ground-based measurements, supporting the dataset’s reliability for anomaly detection.

---

## Project Workflow
The full pipeline followed these stages:

1. **Data Acquisition**
   - Loaded NASA POWER daily climate dataset
   - Organised multicity time-series data

2. **Data Inspection**
   - Checked shape, datatypes, duplicates, and missing values
   - Confirmed dataset completeness and consistency

3. **Preprocessing**
   - Converted date fields
   - Sorted by city and date
   - Cleaned invalid rows
   - Prepared final working dataset

4. **Exploratory Data Analysis**
   - Seasonal temperature trends
   - Variable distributions
   - Correlation analysis
   - Rainfall and humidity analysis across cities

5. **Feature Engineering**
   - Deseasonalised temperature features
   - Absolute temperature deviation
   - Temperature range
   - Rolling mean and rolling standard deviation features
   - Lag features
   - Cyclical time encodings

6. **Time-Based Data Splitting**
   - 80% training
   - 10% validation
   - 10% testing

7. **Scaling**
   - StandardScaler fitted on training data only
   - Applied consistently to validation and test data

8. **Model Development**
   - Baseline Isolation Forest
   - Tuned Isolation Forest
   - Baseline LSTM Autoencoder
   - Tuned LSTM Autoencoder

9. **Evaluation**
   - Train anomaly percentage
   - Validation anomaly percentage
   - Test anomaly percentage
   - Train–Validation gap
   - Train–Test gap
   - Separation Score
   - Threshold (for LSTM)

10. **Final Model Ranking**
    - Models ranked using anomaly separation and stability

---

## Models Used

### 1. Isolation Forest
Isolation Forest was used as the classical unsupervised benchmark model. It identifies anomalies by isolating unusual observations with fewer partitioning steps.

#### Baseline Configuration
- `n_estimators = 200`
- `contamination = 0.05`
- `max_samples = auto`
- `max_features = 1.0`
- `random_state = 42`

#### Tuning
A grid search was used across:
- `n_estimators`
- `contamination`
- `max_samples`
- `max_features`

A total of **36 trials** were evaluated.

---

### 2. LSTM Autoencoder
The LSTM Autoencoder was used to capture temporal dependencies in climate sequences. It reconstructs normal patterns and identifies anomalies using reconstruction loss.

#### Baseline Configuration
- Sequence length: `30`
- Encoder: `LSTM(64)`
- Decoder: `RepeatVector + LSTM(64)`
- Loss: `MSE`
- Optimiser: `Adam`
- Epochs: `50`
- Batch size: `32`
- Early stopping patience: `8`
- Threshold: `95th percentile` of training reconstruction loss

#### Tuned Configuration
Three tuned candidate architectures were tested with variations in:
- LSTM units
- Dropout
- Batch size
- Epochs
- Early stopping patience

The best tuned model was selected using the **lowest validation reconstruction loss**.

---

## Final Results

### Isolation Forest Results

| Metric | Baseline IF | Tuned IF |
|--------|-------------:|---------:|
| Train % | 5.0000 | 3.0060 |
| Validation % | 2.8194 | 1.0573 |
| Test % | 14.6256 | 9.2511 |
| Train–Validation Gap | 2.1806 | 1.9487 |
| Train–Test Gap | 9.6256 | 6.2451 |
| Separation Score | 0.0775 | 0.0879 |

### LSTM Autoencoder Results

| Metric | Baseline LSTM | Tuned LSTM |
|--------|---------------:|-----------:|
| Train % | 5.0025 | 5.0025 |
| Validation % | 17.7778 | 18.2828 |
| Test % | 16.0606 | 10.9091 |
| Train–Validation Gap | 12.7753 | 13.2803 |
| Train–Test Gap | 11.0581 | 5.9066 |
| Separation Score | 0.2910 | 0.2941 |
| Threshold | 0.576421 | 0.630495 |

---

## Final Model Ranking

| Rank | Model | Separation Score | Train–Test Gap |
|-----:|-------|-----------------:|---------------:|
| 1 | Tuned LSTM | 0.2941 | 5.9066 |
| 2 | Baseline LSTM | 0.2910 | 11.0581 |
| 3 | Tuned IF | 0.0879 | 6.2451 |
| 4 | Baseline IF | 0.0775 | 9.6256 |

### Best Model
The **Tuned LSTM Autoencoder** was selected as the final best model because it achieved:
- the **highest Separation Score**
- improved **generalisation**
- a lower **Train–Test Gap** than the baseline LSTM
- better detection of time-dependent climate anomalies

---

## Key Findings
- Deep learning models outperformed classical anomaly detection models on this climate time-series task.
- LSTM-based models captured temporal and seasonal patterns more effectively than Isolation Forest.
- Model tuning improved stability and reduced over-detection on unseen test data.
- Climate anomalies were not evenly distributed across cities and months.
- The strongest anomalies were concentrated mainly in **London** and **Bristol**.
- Late-year months, especially **November** and **December**, showed higher anomaly activity.
