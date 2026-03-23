import pickle
import numpy as np

# Load model
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input
sample = np.array([[5, 7, 80, 65]])  # hours, sleep, attendance, previous

prediction = model.predict(sample)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")