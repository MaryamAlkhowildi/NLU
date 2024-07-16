---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://drive.google.com/drive/folders/1EEMnR63W0d2fwkyg84Jq362zNDYov6k4?usp=drive_link

---

# Model Card for e37076ka-p01679ma-NLI

<!-- Provide a quick summary of what the model is/does. -->

This SVM model is trained to classify text pairs in the context of Natural Language Inference (NLI), determining if a hypothesis is true based on a given premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses a Support Vector Machine (SVM) with a linear kernel, optimized for binary text classification tasks. It has been trained on a dataset comprising premise-hypothesis pairs, utilizing a TF-IDF vectorization approach for text representation.

- **Developed by:** Khawla Almarzooqi and Maryam Alkhowildi
- **Language(s):** English
- **Model type:** Supervised Machine Learning
- **Model architecture:** SVM with Linear Kernel
- **Finetuned from model [optional]:** Custom SVM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** NA
- **Paper or documentation:** NA

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K pairs of premises and hypotheses from the provided NLI dataset were used. Data cleaning involved removing non-alphanumeric      characters and converting all text to lowercase to standardize the input for TF-IDF vectorization.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - Kernel: linear
      - Regularization: Default settings of SVC in scikit-learn

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 6 minutes
      - model size: 41KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Approximately 6K pairs used for development testing that was provided.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision: Measures the quality of the model's positive predictions.
      - Recall: Captures how well the model identifies actual positives.
      - F1-score: Balances precision and recall.
      - Accuracy: Reflects the overall correctness of the model.
      - Confusion Matrix: Further clarify model performance upon each class.

### Results

      - Precision: 0.57 (macro avg)
      - Recall: 0.57 (macro avg)
      - F1-score: 0.57 (macro avg)
      - Accuracy: 58%
      - Confusion Matrix: 
            - True Positives (TP): 2211
            - True Negatives (TN): 1668
            - False Positives (FP): 1591
            - False Negatives (FN): 1267

The model shows a balanced performance across both classes with a slight bias towards more frequent classes.

## Technical Specifications

### Hardware


      - RAM:  at least 8 GB
      - Storage: at least 30GB,
      - No specific GPU requirement

### Software


      - Environment: Google Colab
      - Python: 3.10.12
      - Libraries:
        - Scikit-learn: 1.2.2 - Used for SVM and TF-IDF Vectorizer.
        - Pandas: 2.0.3 - Used for data manipulation.
        - Numpy: 1.25.2 - Used for numerical operations.
        - Seaborn: 0.13.1 - Used for data visualization.
        - Matplotlib: 3.7.1 - Used for data visualization.
        - Joblib: 1.4.0 - Used for model saving and loading.
        - os: Standard library for interacting with the operating system, included with Python.
        - re: Standard library for regular expression operations, included with Python.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model's performance is influenced by the characteristics of the training data, which may contain inherent biases. These biases could skew the model's decision-making, leading to systematic preference for certain outcomes that may not accurately reflect all real-world scenarios. To ensure fairness and reliability, it's crucial to evaluate the model against a more diverse and comprehensive set of data samples. Doing so would help identify any underlying biases and develop strategies to mitigate their effects, thereby enhancing the model's ability to generalize.


## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The choice of hyperparameters and model configuration was based on preliminary tests and evaluations aimed at optimizing classification performance on the development set, which lead to mostly choosing the default settings of the model because of the increased accuracy compared to the baseline model. 
