import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the tennis dataset from a CSV file
df = pd.read_csv('/content/drive/MyDrive/FDSA PROGRAM/DATASET/tennisdata.csv')

# Ensure the required columns are present
required_columns = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'PlayTennis']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in the dataset.")

# Convert categorical variables into numerical values
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Windy'] = df['Windy'].map({'Weak': 0, 'Strong': 1})
df['PlayTennis'] = df['PlayTennis'].map({'No': 0, 'Yes': 1})

# Separate features and target variable
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Handle NaN values in the dataset
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the test set and predictions
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(test_results)
