# Electricity Theft Detection Using Machine Learning

A machine learning-based system for detecting potential electricity theft from smart meter consumption patterns. The project uses behavioral features extracted from historical electricity consumption data and compares Decision Tree, Random Forest, and XGBoost classifiers under class imbalance.

## IEEE Publication

This project is associated with the following IEEE conference publication:

**Behavior-Driven Electricity Theft Detection Using Machine Learning Under Imbalanced Smart Meter Data**

* **Conference:** TQCEBT 2026
* **Publisher:** IEEE
* **Publication:** April 2026
* **IEEE Xplore Document ID:** 11681443
* **DOI:** 10.1109/TQCEBT67648.2026.11681443
* **Pages:** 1–6

**IEEE Xplore:** https://ieeexplore.ieee.org/document/11681443


## Project Overview

Electricity theft creates significant financial losses for power distribution companies and is difficult to identify through manual inspection alone.

This project develops a data-driven approach that analyzes historical electricity consumption behavior and identifies users exhibiting patterns associated with potential electricity theft.

The workflow includes:

* Data preprocessing and missing-value handling
* Temporal train-test splitting
* Consumption behavior analysis
* Feature engineering
* Imbalanced classification
* Decision Tree, Random Forest, and XGBoost modeling
* Threshold optimization for F1-score
* ROC-AUC and PR-AUC evaluation
* Model persistence using Joblib
* Interactive prediction for individual users

## Dataset

The dataset contains daily electricity consumption records for:

* **42,372 consumers**
* **1,034 daily consumption readings**
* Approximately **36.6 million consumption observations after reshaping**
* **3,615 labeled theft consumers**
* Theft prevalence in the original dataset: approximately **8.53%**

The original data contains a strong class imbalance, making accuracy alone an insufficient evaluation metric.

> The raw dataset is not included in this repository. Obtain the dataset from its authorized source and place it in the project directory as `data.csv`.

## Methodology

### 1. Data Preprocessing

The original wide-format smart meter dataset is transformed into a user-date-consumption format.

Missing consumption values are handled using:

* Linear interpolation
* Forward filling
* Positive median fallback

A temporal split is then performed to separate historical consumption from future observations.

### 2. Feature Engineering

Six behavioral features are extracted for each consumer:

| Feature      | Description                                 |
| ------------ | ------------------------------------------- |
| `mean_cons`  | Mean electricity consumption                |
| `std_cons`   | Standard deviation of consumption           |
| `cv`         | Coefficient of variation                    |
| `zero_ratio` | Ratio of zero-consumption readings          |
| `max_diff`   | Maximum change between consecutive readings |
| `event_rate` | Rate of anomalous consumption events        |

A z-score based approach is used to identify abnormal consumption events.

### 3. Machine Learning Models

Three classifiers are evaluated:

* Decision Tree
* Random Forest
* XGBoost

Class imbalance is addressed using class weighting for the tree-based models and `scale_pos_weight` for XGBoost.

### 4. Evaluation

Because electricity theft is a relatively rare event, the project emphasizes:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

The classification threshold is optimized to maximize F1-score rather than relying exclusively on the default 0.5 threshold.

## Results

The final evaluation produced the following results on the test set:

| Model         | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Decision Tree |   0.8119 |    0.8314 | 0.9556 | 0.8892 |  0.7103 | 0.8761 |
| Random Forest |   0.8131 |    0.8294 | 0.9610 | 0.8903 |  0.7287 | 0.8915 |
| XGBoost       |   0.8107 |    0.8288 | 0.9582 | 0.8888 |  0.7088 | 0.8831 |

The Random Forest model achieved an F1-score of **0.8903** and PR-AUC of **0.8915** on the evaluated test set.

### Random Forest Confusion Matrix

```text
                 Predicted
              Normal   Theft
Actual Normal    460     1324
Actual Theft     261     6430
```

The model achieved a recall of **0.9610** for the theft class, indicating that the selected threshold prioritizes detection of potential theft cases.

## Feature Importance

The Random Forest model identified the following feature importance values:

| Feature      | Importance |
| ------------ | ---------: |
| `mean_cons`  |     0.2832 |
| `zero_ratio` |     0.2079 |
| `cv`         |     0.1475 |
| `event_rate` |     0.1326 |
| `std_cons`   |     0.1308 |
| `max_diff`   |     0.0980 |

These values represent model-derived feature importance and should not be interpreted as causal relationships.

## Project Pipeline

```text
Smart Meter Data
       ↓
Data Preprocessing
       ↓
Temporal Train/Test Split
       ↓
Anomaly Detection
       ↓
Behavioral Feature Engineering
       ↓
Class-Imbalanced Training
       ↓
Decision Tree / Random Forest / XGBoost
       ↓
Threshold Optimization
       ↓
Model Evaluation
       ↓
Saved Random Forest Model
       ↓
Individual User Prediction
```

## Interactive Prediction

The notebook also contains an inference function that accepts six behavioral features:

```text
Mean Consumption
Standard Deviation
Coefficient of Variation
Zero Consumption Ratio
Maximum Consumption Difference
Event Rate
```

The saved Random Forest model generates:

* Theft probability
* Classification result
* Decision threshold

Example:

```text
Prediction        : Theft Detected
Theft Probability : 0.5642
Decision Threshold: 0.177
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib
* Jupyter Notebook

## Repository Contents

```text
ETD.ipynb
requirements.txt
saved_models/
src/
results/
```

## Future Improvements

* Develop a complete preprocessing and inference pipeline
* Add cross-validation with time-aware validation
* Perform systematic hyperparameter optimization
* Add explainable AI techniques such as SHAP
* Develop a Flask/FastAPI prediction API
* Build a web dashboard for utility operators
* Evaluate the approach on additional smart-meter datasets



## Author

**Prasanth Kumar Chinta**

Computer Science and Engineering

Research interests: Machine Learning, Data Science, Smart Grid Analytics

## License

This project is intended for academic and research purposes.
