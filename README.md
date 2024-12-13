# Credit Card Fraud Detection

This repository contains my first AI project: a supervised machine learning model that detects fraudulent credit card transactions. It demonstrates the use of Python and machine learning algorithms to analyze financial data and identify potential fraud.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Usage](#setup-and-usage)
- [Dataset](#dataset)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

Fraud detection is a critical task in the financial industry. This project uses a Decision Tree Classifier to analyze a dataset of transactions and predict fraudulent ones. It’s a beginner-friendly introduction to AI, showcasing the application of data processing, visualization, and machine learning.

## Features

- Exploratory Data Analysis (EDA) to understand patterns.
- Correlation analysis to identify key features.
- Implementation of a supervised learning model for classification.
- Model evaluation to assess performance.

## Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Plotly

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fraud-detection
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python scripts/fraud_detection.py
   ```

## Dataset

The dataset used is this project took it from kaggle platform for another project.
after instaling Kaggle, generate the API in account setting and then place the kaggle.json file in path:

- On Windows: C:\Users\<YourUsername>\.kaggle\kaggle.json
- on macOS/Linux: ~/.kaggle/kaggle.json

in root directory of your project Download the Dataset from kaggle with command:

```bash
mkdir data
cd data
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

## Results

The model achieves a high accuracy in detecting fraudulent transactions. It uses features like transaction type, amount, and balances to make predictions.

## Future Improvements

- Use advanced models like Random Forest or Gradient Boosting.
- Incorporate additional features like location or device data.
- Implement real-time fraud detection.
