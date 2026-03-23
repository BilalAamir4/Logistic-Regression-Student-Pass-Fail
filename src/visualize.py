import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocess import preprocess

# Load model
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

X, y = preprocess('../data/student_scores.csv')

# Loss Curve
plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("../results/loss_curve.png")
plt.clf()

# Confusion Matrix
preds = model.predict(X)

tp = np.sum((preds == 1) & (y == 1))
tn = np.sum((preds == 0) & (y == 0))
fp = np.sum((preds == 1) & (y == 0))
fn = np.sum((preds == 0) & (y == 1))

cm = np.array([[tp, fp], [fn, tn]])

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("../results/confusion_matrix.png")