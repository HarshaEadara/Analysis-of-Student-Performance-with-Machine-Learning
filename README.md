# Analysis-of-Student-Performance-with-Machine-Learning
This repository contains a machine learning project aimed at analyzing student performance using various algorithms. The project uses student data to predict performance based on multiple attributes, leveraging machine learning models to generate predictions. The goal is to analyze the factors affecting student performance and build a predictive model that can assist in educational decisions.

## Table of Contents
- [Overview](#overview)
- [Key Insights](#key-insights)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Approach](#approach)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
This analysis leverages a dataset that includes various attributes of students, such as their demographic information, study time, and past academic performance, to predict their final grade. Machine learning algorithms, including classification and regression models, are used to build and evaluate predictive models.

**Key Objectives**
- **Data Preprocessing:** Clean and prepare the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA):** Understand the distribution of the data, visualize key relationships, and investigate factors influencing student performance.
- **Modeling:** Apply various machine learning algorithms to predict student performance. Evaluate the models using metrics such as accuracy, precision, recall, and F1-score.
- **Model Evaluation:** Compare the performance of different models and select the best-performing one for prediction.

## Key Insights
- **Study Time and Academic Performance:** A strong positive correlation exists between the amount of time students spend on studying and their final grade. Students who dedicate more time to their studies tend to perform better academically.
- **Impact of Past Academic Performance:** Previous academic performance is one of the most significant factors in predicting a student’s final grade. Students with higher past scores are more likely to achieve better results in their current academic activities.
- **Gender and Performance:** There may be slight differences in performance across genders, with female students showing a marginally higher average grade in some cases, although this can vary depending on the dataset.
- **Model Performance:** After evaluating various models, algorithms such as Random Forest and Logistic Regression performed the best in predicting student grades. These models showed higher accuracy and generalization compared to others like Support Vector Machines and Decision Trees.
- **Feature Importance:** In the Random Forest model, study time, previous scores, and absences were found to be the most important features affecting the prediction of student performance.
  
## Dataset
The dataset used in this analysis consists of student performance dataset collected from Kaggle and avaliable to use in `data` folder. 

The data includes:
- **Demographic information:** Gender, age, etc.
- **Study-related features:** Time spent on study, number of absences, etc.
- **Past academic performance:** Scores on previous tests, assignments, etc.
- **Final grade:** Target variable (dependent variable) indicating the final grade of the student.

### Preprocessing Steps
- Removal of duplicate entries
- Filtering of sparse user-product interactions
- Normalization and preparation for models

## Technologies Used
The project is implemented using:
- **Programming Language:** Python
- **Libraries and Frameworks:**
   - scikit-learn
   - Pandas
   - Matplotlib
   - Seaborn
   - NumPy
- **Jupyter Notebook:** For analysis and visualization

## Approach
The analysis was conducted in the following systematic steps:
1. **Problem Definition:**
   - Clearly defined the problem as predicting student performance based on demographic, behavioral, and academic attributes.
2. **Data Collection:**
   - Used a Kaggle dataset containing information on students, including their grades, parental education levels, and lunch program participation.
3. **Data Preprocessing:**
   - Removed duplicate entries to ensure data consistency.
   - Performed one-hot encoding for categorical variables to prepare them for machine learning algorithms.
   - Normalized numerical features to bring all variables onto the same scale.
4. **Exploratory Data Analysis (EDA):**
   - Conducted statistical analysis and created visualizations (e.g., bar plots, heatmaps) to understand data distribution and relationships.
   - Identified correlations, such as a strong link between low grades and factors like lunch program type and parental education levels.
5. **Feature Engineering:**
   - Applied Principal Component Analysis (PCA) to reduce dimensionality while retaining key information.
6. **Model Training:**
   - Trained predictive models using Logistic Regression and Random Forest to classify and predict student performance.
7. **Model Evaluation:**
   - Evaluated model performance using metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
   - Compared models to determine the best-performing one for predicting grades.
8. **Insights Extraction:**
   - Highlighted important features like lunch type and parental education, which showed significant correlations with lower student grades.

## Results and Evaluation
### Performance of Models
The performance of the models in predicting student performance was evaluated using various statistical techniques and visualizations. Through Exploratory Data Analysis (EDA), several key variables were identified as having significant correlations with student grades.
 
### Key Findings
- **Lunch Type:** Students receiving free or reduced-price lunch tended to achieve lower grades compared to those who paid for their lunches. This suggests that access to nutritious meals may play a crucial role in academic success.
- **Parental Education Level:** Students whose parents had lower levels of education were more likely to experience poorer academic outcomes. This highlights the importance of parental involvement and support in education.

The radial visualization further supported these findings by clearly showing the relationships between these variables and student performance. The clustering of data points indicated that certain combinations of factors, such as lunch type and parental education level, were more strongly associated with specific performance outcomes.

### Implications
These findings have important implications for educational policy and practice. They suggest that:
- **Nutritional Interventions:** Providing nutritious meals could significantly enhance student performance, particularly for those receiving free or reduced-price lunches.
- **Parental Support Programs:** Initiatives aimed at increasing parental involvement and support, especially for families with lower levels of education, could help improve academic outcomes and reduce achievement gaps.

### Conclusion
Overall, the analysis underscores the need for targeted interventions to address lunch disparities and provide additional support to families with lower levels of education. By implementing these strategies, educational institutions can potentially improve student performance and promote equity in academic achievement.

## Usage
To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Analysis-of-Student-Performance-with-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Analysis-of-Student-Performance-with-Machine-Learning
   ```
3. Install Dependencies:
Make sure you have Python installed. Then install the required libraries:
   ```bash
   pip install requirements.txt
   ```
4. Run the Notebook:
Open the Jupyter Notebook and execute the cells
   ```bash
   jupyter notebook Analysis_of_Student_Performance_with_Machine_Learning.ipynb
   ```
5. Ensure the dataset `StudentsPerformance.csv` is available in the project directory.
6. Run the cells sequentially to execute the analysis.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, make changes, and submit a pull request. Please ensure your code adheres to the project structure and is well-documented.
