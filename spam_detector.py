import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 4. Convert text to numeric features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train classifier
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6. Accuracy
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 7. Test user input
while True:
    msg = input("\nEnter a message (or 'exit' to stop): ")
    if msg.lower() == 'exit':
        break

    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]

    if pred == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")
