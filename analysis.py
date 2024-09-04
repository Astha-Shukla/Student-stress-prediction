#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read data from CSV file
data = pd.read_csv("StressLevelDataset.csv")
data.head(20)

data.tail()

data.shape

data[data['mental_health_history'] == 1]

data.duplicated().sum()

data.isnull().sum()

data[(data['bullying'] < 2)]

corelation = data.corr()
plt.figure(figsize=(14,10))
sns.heatmap(data = corelation,annot = True)
plt.title("What tends to higher stress level")
plt.show()

physioligical_factor = data[['headache','blood_pressure', 'sleep_quality', 'breathing_problem', 'stress_level']]
correlation_physiological = physioligical_factor.corr()
plt.figure(figsize=(10,8))
sns.heatmap(data=correlation_physiological, annot=True)
plt.title('Physioligical Factor vs Stress Level')
plt.show()


# Extract input features (mental_heealth, depression,blood_pressure,sleep_quality,breathing_problem,noise_level,study_load,extraxurricular_activities) and SOC values
X = data[['mental_health_history','depression','blood_pressure','sleep_quality','breathing_problem','noise_level','study_load','extracurricular_activities']].values
stress = data['stress_level'].values
print(stress)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, stress, test_size=0.2, random_state=42)
print(y_test)

len(y_test)

# Train a Random Forest regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)


#Make predictions on the testing set
stress_predictions = random_forest_model.predict(X_test)
print(stress_predictions)


from sklearn.metrics import mean_squared_error


# Model evaluation
mse = mean_squared_error(y_test, stress_predictions)
r2 = r2_score(y_test, stress_predictions)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Plot the true stess values vs. predicted stress values
plt.scatter(y_test, stress_predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Diagonal line
plt.xlabel('True Stress')
plt.ylabel('Predicted stress')
plt.title('True Stress vs. Predicted stress (Random Forest)')
plt.grid(True)
plt.show()


predictions_df = pd.DataFrame({
    'Actual state of charging': (y_test),# current load other name state of health

    'Predicted state of charging ': (stress_predictions)
})

# Display the DataFrame
print(predictions_df)


print(y_test)


print(stress)


#Convert predictions and true labels to binary format
# For example, let's assume we're interested in binary classification where 1 is positive and 0 is negative
threshold = 0.5  # You can adjust this threshold based on your problem
y_pred_binary = [1 if pred >= threshold else 0 for pred in stress_predictions]
y_true_binary = [1 if true >= threshold else 0 for true in y_test]



print(y_pred_binary)


print(y_true_binary)



from sklearn.metrics import confusion_matrix, f1_score


cm = confusion_matrix(y_true_binary,y_pred_binary )

print("Confusion Matrix:")
print(cm)

import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt


# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()


from sklearn.metrics import f1_score


# Calculate F1 score
f1 = f1_score(y_true_binary, y_pred_binary, average='weighted')
print("F1 Score:", f1)



# Plot F1 score
plt.figure(figsize=(8, 6))
plt.bar(["Random Forest"], [f1], color='skyblue')
plt.title("F1 Score Comparison")
plt.xlabel("Algorithm")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()




