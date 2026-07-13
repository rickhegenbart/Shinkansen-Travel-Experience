# Shinkansen Passenger Experience Prediction

An end-to-end machine learning project that predicts whether a passenger will report a positive overall experience when traveling on Japan's Shinkansen bullet train system.

The project combines passenger travel information with survey-based service ratings, compares multiple classification models, and identifies the factors most strongly associated with passenger satisfaction. The final CatBoost model achieved **95.72% validation accuracy** and an **F1 score of 0.9605**.

## Project Objective

The target variable is `Overall_Experience`:

- `1` — Positive passenger experience
- `0` — Negative passenger experience

The project addresses two questions:

1. How accurately can passenger satisfaction be predicted?
2. Are operational factors, such as delays and travel distance, or service-quality factors, such as comfort and entertainment, more influential?

## Dataset

The notebook uses four CSV files:

```text
Traveldata_train.csv
Surveydata_train.csv
Traveldata_test.csv
Surveydata_test.csv
```

The travel and survey files are merged using the shared `ID` column.

| Dataset | Records | Description |
|---|---:|---|
| Training data | 94,379 | Travel and survey features with the target |
| Test data | 35,602 | Travel and survey features without the target |
| Raw model features | 23 | Numeric and categorical passenger attributes |
| Engineered features | 35 | Raw features plus delay, distance, and interaction features |

The source data files are not included in this repository. Add them to Google Drive or update the notebook's file paths before running it.

## Project Workflow

1. Load and merge travel and passenger survey data.
2. Check duplicate records, data types, and missing values.
3. Explore target balance and feature relationships.
4. Split the data into stratified training and validation sets.
5. Apply leakage-safe feature engineering.
6. Build preprocessing pipelines for numeric and categorical variables.
7. Compare baseline and advanced classification models.
8. Tune HistGradientBoosting and XGBoost.
9. Train and evaluate CatBoost using native categorical features.
10. Perform five-fold stratified cross-validation.
11. analyze feature importance and business implications.
12. Train the final model and generate `Final_Submission.csv`.

## Feature Engineering

The custom `ShinkansenFeatureEngineer` learns transformation rules from training data and applies the same rules to validation and test data.

Engineered features include:

- Missing-delay indicators
- Total delay
- Difference between arrival and departure delays
- Delay ratio
- Departure and arrival delay flags
- Training-derived travel-distance groups
- Travel class and customer type interaction
- Travel class and travel purpose interaction
- Seat comfort and entertainment interaction
- Online support and booking interaction

This design reduces data leakage and maintains consistent feature structures across datasets.

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting
- XGBoost
- Soft Voting Ensemble
- CatBoost

## Model Performance

| Model or Stage | Validation Accuracy | F1 Score |
|---|---:|---:|
| CatBoost | **95.72%** | **0.9605** |
| Tuned XGBoost | 95.62% | 0.9596 |
| Voting Ensemble | 95.59% | 0.9593 |
| Tuned HistGradientBoosting | 95.43% | 0.9578 |
| Best full-experience baseline | 94.92% | 0.9532 |
| Best operational-only model | 79.14% | 0.8183 |

CatBoost was selected because it produced the strongest validation performance and handles categorical variables directly.

### Cross-Validation Results

Five-fold stratified cross-validation was performed with feature engineering and data preparation refitted inside each fold.

| Metric | Result |
|---|---:|
| Mean accuracy | 95.74% |
| Accuracy standard deviation | 0.00144 |
| Mean F1 score | 0.9607 |
| F1 standard deviation | 0.00131 |
| Mean best iteration | 700 |

The low variation across folds indicates stable model performance.

## Key Findings

The strongest predictors of passenger experience were:

1. Seat comfort
2. Type of travel
3. Seat-comfort and entertainment interaction
4. Travel-class and customer-type interaction
5. Ease of online booking
6. Platform location
7. Onboard entertainment
8. Gender
9. Travel-class and travel-purpose interaction
10. Arrival-time convenience

The operational-only model reached approximately 79% accuracy, while models that included passenger survey and service-quality information reached approximately 95–96%. This suggests that comfort, service quality, booking experience, and onboard conditions are more predictive of satisfaction than delays and travel distance alone.

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- CatBoost
- Google Colab

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```

### 3. Add the data files

Place the four required CSV files in an accessible folder. The current notebook expects them in the root of Google Drive:

```python
travel_train = pd.read_csv("/content/drive/MyDrive/Traveldata_train.csv")
survey_train = pd.read_csv("/content/drive/MyDrive/Surveydata_train.csv")
travel_test = pd.read_csv("/content/drive/MyDrive/Traveldata_test.csv")
survey_test = pd.read_csv("/content/drive/MyDrive/Surveydata_test.csv")
```

Update these paths when using a different directory or local environment.

### 4. Run the notebook

Open `Shinkansen_Travel_Experience_Complete.ipynb` in Google Colab or Jupyter Notebook and run all cells in order.

The final predictions are saved as:

```text
Final_Submission.csv
```

The output contains:

```text
ID,Overall_Experience
```

## Business Recommendations

- Prioritize seat comfort, spacing, and cleanliness.
- Improve onboard entertainment and service consistency.
- Strengthen online booking, online support, and passenger communication.
- Improve platform accessibility, wayfinding, and arrival convenience.
- Evaluate satisfaction across the full passenger journey rather than focusing only on delays.
- Use an operational-only model when survey responses are unavailable in real time.

## Limitations

- Several high-value predictors are survey responses that may only be available after a trip.
- Global feature importance does not explain individual predictions.
- Additional fairness, drift, and production-monitoring checks would be required before deployment.
- Interaction features should be monitored for overfitting.
- The notebook generates predictions but does not include a deployed application or API.

## Future Improvements

- Add SHAP-based local and global model explanations.
- Create a real-time operational model using only pre-trip and in-trip features.
- Add fairness analysis across passenger groups.
- Track model and data drift.
- Package the pipeline for reproducible local execution.
- Deploy the model through an API and interactive dashboard.

## Repository File

```text
Shinkansen_Travel_Experience_Complete.ipynb
```

## Author

**Rick Hegenbart**
