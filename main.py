import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"C:\Users\dell\Desktop\data_for_weka_aw.csv")

# Setting the visual style
sns.set(style='darkgrid')

# Boxplot for heart rate by activity type
plt.figure(figsize=(10, 6))
sns.boxplot(x='activity_trimmed', y='Applewatch.Heart_LE', data=df)
plt.title('Heart Rate by Activity')
plt.ylabel('Heart Rate')
plt.xlabel('Activity Type')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for steps vs heart rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Applewatch.Steps_LE', y='Applewatch.Heart_LE', hue='activity_trimmed', palette='deep', data=df)
plt.title('Steps vs Heart Rate')
plt.xlabel('Steps')
plt.ylabel('Heart Rate')
plt.legend(title='Activity Type')
plt.show()

# Histograms for steps and heart rate
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Applewatch.Steps_LE'], bins=10, kde=True)
plt.title('Histogram of Steps')
plt.xlabel('Steps')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['Applewatch.Heart_LE'], bins=10, kde=True)
plt.title('Histogram of Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



heart_rate = df['Applewatch.Heart_LE']
calories = df['Applewatch.Calories_LE']

corr, _ = pearsonr(heart_rate, calories)
print("Pearson correlation is:", corr)

# Independent variable = heart rate
# Dependent variable = calories
X = heart_rate.values.reshape(-1, 1)  # Independent variable (Heart Rate)
y = calories.values  # Dependent variable (Calories)

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict calories based on heart rate
predicted_calories = model.predict(X)

# Plot the original data
plt.scatter(heart_rate, calories, color='blue', label='Actual Calories', s=5)

# Plot the regression line (Predicted calories)
plt.plot(heart_rate, predicted_calories, color='red', label='Predicted Calories')

plt.xlabel('Heart Rate')
plt.ylabel('Calories')
plt.title('Heart Rate vs Calories with Regression Line')
plt.legend()
plt.show()

# Print the model's slope and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
