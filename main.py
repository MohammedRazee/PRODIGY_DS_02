
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
gender_submission_df = pd.read_csv('./data/gender_submission.csv')

# Copy of train data for processing
df = train_df.copy()

# DATA CLEANING

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin, Ticket, and Name
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# Convert categorical columns
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Create FamilySize column
df['FamilySize'] = df['SibSp'] + df['Parch']




# EXPLORATORY DATA ANALYSIS

sns.set_theme(style='whitegrid')
plt.figure(figsize=(12, 8))

# 1. Survival Count
plt.subplot(2, 3, 1)
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.xticks([0, 1], ['Died', 'Survived'])

# 2. Survival by Sex
plt.subplot(2, 3, 2)
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.legend(['Died', 'Survived'])

# 3. Survival by Pclass
plt.subplot(2, 3, 3)
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.legend(['Died', 'Survived'])

# 4. Age Distribution
plt.subplot(2, 3, 4)
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')

# 5. Survival by Age
plt.subplot(2, 3, 5)
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age vs Survival')

# 6. Correlation Heatmap
plt.subplot(2, 3, 6)
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()
