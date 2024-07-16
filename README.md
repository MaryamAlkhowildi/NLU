# Project Overview

This project is part of the Natural Language Understanding coursework. It involves developing two solution for pairwise sequence classification, specifically focusing on NLI(Natural Language Inference). This README outlines the operational details of the demo and training notebooks designed to run in Google Colab.

## File Structure

- Folder `SVM Solution Category A`
    - `Demo_code_SVM.ipynb`: Jupyter notebook containing the code to load the SVM pre-trained model, preprocess input data, generate predictions, and save them to a CSV file.
    - `SVM.ipynb`: Jupyter notebook containing the code used for training and evaluating the SVM model on the training and development datasets.
    - `svm_model.joblib`: Saved SVM model trained on the NLI dataset.
    - `tfidf_vectorizer.joblib`: Saved TF-IDF vectorizer used for preprocessing the text data.
    - `SVM_NLI_model_card.md`: Model card for SVM model used.
    - `Group_3_A.csv`: Predictions made by the SVM model on the test data.

- Folder `Bi-LSTM Solution Category B`
    - `Demo_code_LSTM.ipynb`: Jupyter notebook containing the code to load the Bi-LSTM pre-trained model, preprocess input data, generate predictions, and save them to a CSV file.
    - `Bi_LSTM.ipynb`: Jupyter notebook containing the code used for training and evaluating the Bi-LSTM model on the training and development datasets.
    - `Bi-lstm_model.h5`: Saved Bi-LSTM model trained on the NLI dataset.
    - `tokenizer.joblib`: Saved tekonizer used for preprocessing the text data.
    - `Bi-lstm_NLI_model_card.md`: Model card for Bi-LSTM model used.
    - `Group_3_B.csv`: Predictions made by the Bi-LSTM model on the test data.

## Usage Instructions

- Ensure all dependencies are installed within Google Colab session for the demo code, as instructed by the notebook.
- Upload the svm_model.joblib, tfidf_vectorizer.joblib, Bi-lstm_model.h5, tokenizer.joblib and test dataset to Google Colab session, in order to proceed with the demo code Notebook.

## Running the Notebook

- Open Demo_code.ipynb in Google Colab.
- Replace the placeholder filename with the name of the actual test data file.
- Run all cells in the notebook sequentially to generate predictions.

## Output

- The predictions are saved to Group_3_A.csv and Group_3_B.csv, located in left-hand side file section of the Google Colab session.
- The notebook also prints the first five predictions for quick verification.

## Troubleshooting

- If encounter errors related to missing files, please ensure that all files are uploaded to the Google Colab session and correctly named.
- For issues with missing packages, ensure all dependencies packages are installed in your Colab environment.

## Project Access

- All work related to this coursework, including models trained, and documentation, is available through this Google Drive link: https://drive.google.com/drive/folders/1EEMnR63W0d2fwkyg84Jq362zNDYov6k4?usp=drive_link 