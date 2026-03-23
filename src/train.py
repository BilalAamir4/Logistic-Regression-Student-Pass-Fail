from preprocess import preprocess
from model import LogisticRegression
import pickle

X, y = preprocess('../data/student_scores.csv')

model = LogisticRegression(lr=0.01, epochs=1000)
model.fit(X, y)

# Save model
with open('../model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training complete. Model saved.")