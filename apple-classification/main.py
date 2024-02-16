import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('apple_quality.csv')
sns.pairplot(df, hue='Quality', diag_kind='kde')
plt.suptitle('Pairplot of Features by Quality', y=1.02)
plt.show()

for column in df.columns[1:-1]:  
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Quality', y=column, data=df)
    plt.title(f'Box Plot of {column} by Quality')
    plt.show()
    
X = df.drop(['A_id', 'Quality'], axis=1)
y = df['Quality']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()