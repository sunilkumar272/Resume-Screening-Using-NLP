# Resume-Screening-Using-NLP

## Description
This is a resume screening project using Natural Language Processing (NLP) techniques. The project aims to classify 962 resumes into 25 different job categories. Four different classification models are implemented and evaluated, including OneVsRest KNeighbors, Multinomial Naive Bayes, OneVsRest with Random Forest, and OneVsRest with Support Vector Machine (SVM). The project uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to preprocess and represent the resume data for classification.

## Dataset
The dataset is downloaded from Kaggle. The dataset used in this project includes 962 resumes, each associated with one of the 25 job categories. The dataset serves as the foundation for training and evaluating the classification models. It is a critical component of the project and allows for the assessment of the model's performance in categorizing resumes based on job categories.

Link to the Dataset:
https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

## Results
KNeighbors model outperforms the other models in terms of mean accuracy. This superior performance can be attributed to the characteristics of the KNeighbors classifier, which is effective in capturing the underlying patterns in the data. Random Forest and Support Vector Machine (SVM) models show very high accuracy scores, but they exhibit signs of overfitting as indicated by their perfect scores in some cross-validation folds. The choice of model should take into account both the mean accuracy and the potential for overfitting. The KNeighbors model strikes a balance between accuracy and generalization, making it a suitable choice for this resume screening task.
