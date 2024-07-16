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

This Bidirectional LSTM model is trained to classify text pairs in the context of Natural Language Inference (NLI), determining if a hypothesis is true (entailment), false (contradiction), or undetermined (neutral) based on a given premise.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses a Bidirectional Long Short-Term Memory (Bi-LSTM) network, optimized for sequence data and capable of understanding context from both directions, which is crucial for NLI tasks. It has been trained on a dataset comprising premise-hypothesis pairs, utilizing a tokenizer for text preprocessing and embedding layers for text representation.

- **Developed by:** Khawla Almarzooqi and Maryam Alkhowildi
- **Language(s):** English
- **Model type:** Supervised Deep Learning
- **Model architecture:** Bidirectional LSTM
- **Finetuned from model [optional]:** Custom Bi-LSTM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** NA
- **Paper or documentation:** NA

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K pairs of premises and hypotheses from the provided NLI dataset were used. Data cleaning involved removing non-alphanumeric characters and converting all text to lowercase to standardize the input. Text data was then tokenized using a Keras tokenizer, which converts text into sequences of integers. These sequences were padded to ensure uniform input size for the LSTM model. A Bidirectional LSTM architecture was employed to enhance the model's ability to understand context by processing text sequences from both directions, significantly improving its performance on sequence-dependent tasks like NLI.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

- **Model Architecture**: Bidirectional LSTM
- **Units per LSTM Layer**: 64 units in each direction of the Bi-LSTM
- **Embedding Dimension**: 100 (Each word is represented by a 100-dimensional vector)
- **Dropout Rate**: 0.5 (To reduce overfitting by randomly setting a fraction of input units to 0 at each update during training)
- **Optimizer**: Adam (A method for stochastic optimization)
- **Loss Function**: Binary Crossentropy (For binary classification tasks like NLI)
- **Batch Size**: 32 (Number of samples per gradient update)
- **Number of Epochs**: 10 (Number of times the learning algorithm will work through the entire training dataset)


#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

- Overall training time: 30 min
- Model size: 45KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Approximately 6K pairs used for development testing.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision: Measures the quality of the model's positive predictions.
      - Recall: Captures how well the model identifies actual positives.
      - F1-score: Balances precision and recall.
      - Accuracy: Reflects the overall correctness of the model.
      - Confusion Matrix: Further clarify model performance upon each class.

### Results

      - Precision: 0.60 (macro avg)
      - Recall: 0.60 (macro avg)
      - F1-score: 0.60 (macro avg)
      - Accuracy: 60%
      - Confusion Matrix:
            - True Positives (TP): 2160
            - True Negatives (TN): 1900
            - False Positives (FP): 1359
            - False Negatives (FN): 1318

The model demonstrates a consistent performance across precision, recall, and F1-score metrics, each achieving 0.60. However, the overall accuracy at 60% indicates there is room for improvement, especially in reducing the number of false positives and false negatives. The confusion matrix provides detailed insights into the model's ability to classify each class correctly, showing relatively balanced true positive and true negative results but with notable misclassifications as indicated by the false positives and false negatives. This highlights potential areas for further model refinement and training with more balanced or additional data to improve its generalization capabilities.

## Technical Specifications

### Hardware

      - RAM:  at least 9 GB
      - Storage: at least 80GB
      - No specific GPU requirement

#### Software

- **Environment**: Google Colab
- **Python**: 3.10.12
- **Libraries**:
  - Keras: 2.8.0 - Used for model building and training.
  - TensorFlow: 2.8.0 - Backend for Keras, used for all neural network operations.
  - Pandas: 2.0.3 - Used for data manipulation.
  - Numpy: 1.25.2 - Used for numerical operations.
  - Seaborn: 0.13.1 - Used for data visualization.
  - Matplotlib: 3.7.1 - Used for data visualization.
  - Joblib: 1.4.0 - Used for saving and loading the tokenizer.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Despite improvements in accuracy and understanding, the Bi-LSTM model may still inherit biases present in the training data, which could affect its performance and fairness. Continuous evaluation on diverse datasets is necessary to identify and mitigate these biases, ensuring the model's robustness and reliability in real-world scenarios.

## Additional Information

The selection of hyperparameters and architecture was carefully tested to optimize the model's performance on the development set. Future work should consider exploring more sophisticated architectures or incorporating external datasets for further performance enhancements.

