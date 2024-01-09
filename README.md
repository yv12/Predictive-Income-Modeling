# Predictive-Income-Modeling
Classification of Income using different Machine Learning Algorithm

1. PROBLEM DEFINITION
• The prevalent issue of economic imbalance in the United States is the current issue. As the top 
1% of earners continue to capture a larger share of total income, many households struggle to 
meet their financial needs. Due to Stagnant Wages a lack of Financial Literacy.
• Incorporating Income patterns can exhibit temporal dynamics due to changes in employment, 
economic conditions, and personal circumstances.
• The challenge is to evaluate and compare machine learning models using the census income 
dataset. The task is to identify effective models for predicting income levels and improve their 
performance. 
• It requires extensive analysis, testing, and evaluation of multiple models, dealing with large 
amounts of data, and understanding complex algorithms

2. SOLUTION STRATEGY
2.1Gather Data:
• Identify the sources of data needed for the project. The main source would be the census 
dataset, which contains information about individuals such as age, education, occupation, 
marital status, etc.
• Dataset is obtained from UCI Machine Learning Repository: Census Income Data Set
archive.ics.uci.edu/ml/datasets/census+income [6].
• Access the census dataset either from a reliable external source or an internal database.
2.2 Data Preprocessing:
• Perform data cleaning to handle missing values, outliers, and inconsistencies in the dataset. 
This may involve techniques such as imputation, removal, or data transformation.
2.3 Exploratory Data Analysis:
• Perform exploratory data analysis to gain insights into the dataset, identify patterns, and 
understand the relationships between variables like age, sex, education, race, relationship, 
Occupation, etc.
• Use visualizations, statistical summaries, and correlation analysis to uncover key insights and 
potential predictors of income.
2.4 Feature Engineering:
• Create new features or modify existing ones that might improve the predictive power of the 
model. This could involve combining related variables, creating dummy variables for 
categorical features, or scaling numerical variables.
2.5 Model Selection and Training:
• Select an appropriate machine learning algorithm(s) for the task, such as logistic regression, 
random forest, or gradient boosting.
• Evaluate the model's performance on the validation set using suitable metrics like accuracy, 
precision, recall, and F1-score.
2.6 Model Evaluation and Tuning:
• Assess the model's performance on the validation set and compare it with other models 
considered.
19
• If necessary, fine-tune the hyperparameters of the chosen model using techniques like grid 
search, random search, or Bayesian optimization.
• Iterate the process of training, evaluation, and tuning until satisfactory performance is 
achieved.
2.7 Final Model Selection and Evaluation:
• Once the best-performing model is identified and tuned, evaluate its performance on the testing 
set to get an unbiased estimate of its effectiveness.
• Compute relevant evaluation metrics and analyze the model's predictions.
2.8 Model Deployment:
• If the model meets the desired performance criteria, prepare it for deployment.
• Save the trained model and necessary preprocessing steps for future use.
• Document the model, including information about its inputs, outputs, limitations, and any 
assumptions made during the development process


3. RESULTS AND DISCUSSION
3.1Experimental Configuration
We show the experimental results acquired from training and assessing various machine learning 
models on the Census Income Data Set in this part. We explore the ramifications of the results and 
provide detailed information about the performance measures.
3.1.1Model Choice
For the income prediction problem, we used a variety of common machine learning methods, 
including decision trees, random forests, logistic regression. Using the necessary parameters and
settings, each algorithm was trained and evaluated.
3.2Experimental Findings
Using a train-test split, we trained and tested the selected machine learning models on the Census 
Income Data Set. As indicated in the technique section, the dataset was preprocessed and encoded. 
After that, the models were trained on the training set and tested on the test set.
3.2.1 Model Performance
The table below summarizes each model's performance on the income prediction task:
Model Accuracy Precision Recall F1-Score AUC-ROC
Decision
Trees 0.801 0.594 0.610 0.602 0.737
KNN 0.764 0.535 0.322 0.402 0.615
Random
Forests
0.834 0.683 0.608 0.644 0.758
Naïve
Bayes
0.778 0.599 0.294 0.394 0.615
Logistic
Regression
0.788 0.689 0.253 0.370 0.608
FNN 0.780 0.796 0.153 0.257 0.570
Table 8.2.1: Model’s Performance
34
8.3 Discussion
The following observations and inferences can be drawn from the experimental results:
1) The random forest model had the highest accuracy (0.834) and showed a good balance 
between precision and recall.
2) The decision tree model performed reasonably well with an accuracy of 0.801.
3) The K-nearest neighbors (KNN), naïve Bayes, logistic regression, and feedforward 
neural network (FNN) models had lower performance metrics.
4) These models had lower accuracy, precision, recall, and F1-score values, indicating a 
lower ability to correctly identify positive instances.
5) The AUC-ROC scores were also relatively lower for these models, suggesting a fair 
discrimination ability.
6) Further analysis and improvement are needed for the KNN, naïve Bayes, logistic 
regression, and FNN models to enhance their performance on the given dataset The
precision, recall, and F1-score metrics illustrate the models' ability to accurately forecast 
positive events (income more than $50,000). In these criteria, random forests
and neural networks beat the other models, suggesting their effectiveness in identifying 
individuals with greater incomes.
7) The AUC-ROC scores demonstrate how well the models can distinguish between 
positive and negative events. The highest AUC-ROC values were reached by random 
forests and Decision Tree suggesting their higher discriminatory power.



![image](https://github.com/yv12/Predictive-Income-Modeling/assets/87942632/bdad4507-a172-4978-80e7-a22fe5cf602b)
