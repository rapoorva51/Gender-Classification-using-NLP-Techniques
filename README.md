# Gender Classification using NLP Techniques

Welcome to the Gender Classification using NLP Techniques project repository! This project aims to classify gender based on text descriptions using Natural Language Processing (NLP) techniques. If you're new to this project, we'll guide you through the tools used and the process undertaken in simple words.

## Project Overview

The goal of this project is to predict the gender (male or female) of individuals based on text descriptions they provide. This is achieved using NLP techniques to analyze the text data.

## Tools Used

- Python: The primary programming language used for data processing, analysis, and modeling.
- Pandas: A Python library for data manipulation and analysis.
- Scikit-learn: A machine learning library that provides tools for data preprocessing, modeling, and evaluation.
- NLTK (Natural Language Toolkit): A library for working with human language data, including tokenization and lemmatization.
- Matplotlib: A library for creating data visualizations.
- Jupyter Notebook: An interactive environment for running Python code and documenting the analysis.

## Project Process

1. **Data Loading and Exploration:** We start by loading the dataset (`gender-classifier-DFE-791531.csv`) and exploring its contents to understand the structure of the data.

2. **Data Cleaning:** We clean the data by removing missing values and converting the gender labels to numerical values (1 for female, 0 for male).

3. **Text Preprocessing:** The text descriptions are preprocessed by removing special characters, converting text to lowercase, tokenizing words, removing stop words, and lemmatizing the words.

4. **Feature Extraction:** We use the Count Vectorizer to convert the preprocessed text data into a numerical format suitable for machine learning.

5. **Exploratory Data Analysis (EDA):** We visualize the distribution of gender in the dataset using pie charts to gain insights.

6. **Model Building:** We build several classification models, including Naive Bayes, k-Nearest Neighbor, Random Forest, Logistic Regression, and Decision Tree, to predict gender based on the text features.

7. **Cross-Validation:** We evaluate model performance using cross-validation techniques and various metrics like accuracy, precision, recall, and F1-score.

8. **Lasso Regression:** We apply Lasso Regression to understand the importance of individual words in predicting gender.

9. **Neural Network:** We build a Perceptron-based and Multilayer Perceptron (MLP) neural network to classify gender.

10. **Final Evaluation:** We provide the accuracy scores for the models, helping us understand how well each model predicts gender based on text descriptions.

## Contributions

Contributions to this project are welcome! If you want to contribute, you can follow these steps:

1. Fork this repository.
2. Create a new branch for your contribution.
3. Make your changes and commit them with clear descriptions.
4. Push your changes to your fork.
5. Create a pull request, explaining the purpose of your contribution and any improvements or additions you made.

We appreciate your contributions to make this project even better!

## License

This project does not have a specific license, meaning you are free to use the code and data for educational and non-commercial purposes. However, please respect data source licenses and provide proper attribution when necessary.

Feel free to contact us if you have any questions or suggestions. Happy coding!🙂
