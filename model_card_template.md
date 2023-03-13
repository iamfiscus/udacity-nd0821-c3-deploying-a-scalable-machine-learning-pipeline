# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model date: 2023-03-05
- Model version: 1.0.0
- Model type: sklearn.ensemble.RandomForestClassifier

## Intended Use
- Submission for project 3 Udacity ML DevOps nanodegree
## Training Data
- [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) by UCI

## Evaluation Data
- Dataset split into 20% for testing and 80% for training
- Data preprocessed with:
    - OneHotEncoder(sparse=False, handle_unknown="ignore") for categorical data
    - Returns: 
        - If salary > 50k = 0
        - If salary <= w50k = 1
## Metrics
- Precision: 0.748 (0.7479119210326499)
- Recall: 0.630 (0.6297953964194374)
- Fbeta: 0.684 (0.6837903505727179)

## Ethical Considerations
- This should not be utilized for other Udacity projects.
- The training data for the model included variables such as marital-status, relationship, race, sex, and native-country. However, the use of these variables could potentially introduce unfair bias into the analysis of the model's output.
- Additionally, it's important to note that the data was sourced from the United States Census Bureau's census, and caution should be exercised when making inferences for data from other countries.
## Caveats and Recommendations
- The training dataset contains missing values marked as '?' for certain attributes, specifically for workclass, occupation, and native-country.