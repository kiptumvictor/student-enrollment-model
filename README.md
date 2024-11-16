# Student Enrollment Prediction

This project aims to predict student enrollment status using various machine learning models. The provided code preprocesses the student data, trains multiple models, evaluates their performance, and identifies the best model for predicting enrollment status.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Features](#features)
- [Privacy Considerations](#privacy-considerations)
- [Results and Visualization](#results-and-visualization)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/student-enrollment-prediction.git
    ```
2. Change directory to the project folder:
    ```bash
    cd student-enrollment-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the project folder and ensure it is named `combined_student_enrollment_data.csv`.
2. Run the Python script to execute the data preprocessing, model training, and evaluation:
    ```bash
    python predict_enrollment.py
    ```

## Models Used

The following machine learning models are trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

The best performing model is saved for future predictions.

## Features

The dataset includes the following features which are used for prediction:

- `student_id`: Unique identifier for each student.
- `program_enrolled`: The program the student is enrolled in.
- `year_enrolled`: The year the student enrolled.
- `year_graduated`: The year the student graduated (if applicable).
- `enrollment_status`: The target variable indicating if the student is currently enrolled.
- `GPA`: The student's grade point average.
- `attendance_rate`: The student's attendance rate.
- `academic_warnings`: Number of academic warnings received.
- `course_completion_rate`: The rate at which the student completes courses.
- `age`: The age of the student.
- `gender`: The gender of the student.
- `socioeconomic_status`: The socioeconomic status of the student.
- `financial_aid_status`: Whether the student receives financial aid.

## Privacy Considerations

To protect student privacy, the following measures are taken:

- Anonymization of personally identifiable information (PII).
- Encryption of data both at rest and in transit.
- Implementation of access controls.
- Compliance with relevant privacy regulations.
- Use of aggregated data where possible.

## Results and Visualization


The code includes functions to visualize the data and model results:

- Distribution of enrollment status.
- Correlation matrix of numerical features.
- Feature importance for the best performing model.

## Saving and Loading the Model

The best performing model is saved to a file for future use:
```python
joblib.dump(best_model, 'student_enrollment_model.pkl').


