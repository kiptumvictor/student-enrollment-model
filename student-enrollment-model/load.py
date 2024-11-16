import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
Victor = pd.read_csv('combined_student_enrollment_data.csv')  # Replace with your dataset path

# Separate features and target variable
X = Victor.drop('enrollment_status', axis=1)
y = Victor['enrollment_status']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Function to extract feature names after preprocessing
def get_feature_names_after_transform(preprocessor, X):
    preprocessor.fit(X)
    cat_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numerical_features, cat_feature_names])
    return all_feature_names

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Basic statistics and visualizations
print("Dataset loaded successfully. Here are the first 6 rows:")
print(Victor.head(6))

# Basic statistics
print(Victor.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='enrollment_status', data=Victor)
plt.title('Enrollment Status Distribution')
plt.show()

# Select only numerical features for the correlation matrix
numerical_data = Victor.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create a pipeline and fit a model
def create_pipeline_and_fit(model):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f'Model: {model.__class__.__name__}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'AUC-ROC: {roc_auc_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    print('-' * 30)
    return pipeline

# Initialize models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()
]

# Train and evaluate models
pipelines = []
for model in models:
    pipeline = create_pipeline_and_fit(model)
    pipelines.append(pipeline)

# Hyperparameter tuning for RandomForestClassifier using GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=pipelines[2], param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_}')

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'student_enrollment_model.pkl')

# Confirm that the model was saved correctly
print("Best model saved as 'student_enrollment_model.pkl'.")

# Load the model (for prediction)
loaded_model = joblib.load('student_enrollment_model.pkl')

# Example prediction
sample_data = X_test.iloc[0:1, :]  # Ensure sample_data is a DataFrame
prediction = loaded_model.predict(sample_data)
print(f'Predicted enrollment status: {prediction}')

# Function to plot feature importance
def plot_feature_importance(model, X):
    feature_names = get_feature_names_after_transform(preprocessor, X)
    importance = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Plot feature importance for the loaded RandomForest model
plot_feature_importance(loaded_model, X)
