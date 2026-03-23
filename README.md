# Logistic Regression – Student Pass/Fail Prediction

## 📌 Overview

This project implements **Logistic Regression from scratch** to predict whether a student will **pass or fail** based on academic and lifestyle factors.

The goal is to demonstrate a clear understanding of:

* Binary classification
* Sigmoid function
* Gradient descent optimization
* Model evaluation techniques

---

## 📂 Dataset

The dataset contains the following columns:

* `student_id`
* `hours_studied`
* `sleep_hours`
* `attendance_percent`
* `previous_scores`
* `exam_score`

### 🎯 Target Variable

A new column is created:

```
pass = 1 if exam_score >= 50 else 0
```

---

##  Features Used

The model uses the following input features:

* Hours studied
* Sleep hours
* Attendance percentage
* Previous scores

---

## 🧠 Algorithm Explanation

This project implements **Logistic Regression manually (without ML libraries)**.

### 1. Sigmoid Function

Used to convert predictions into probabilities:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

---

### 2. Prediction

The model computes:

[
z = XW + b
]

Then applies sigmoid:

[
\hat{y} = \sigma(z)
]

---

### 3. Loss Function (Binary Cross-Entropy)

[
Loss = -\frac{1}{n} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
]

---

### 4. Optimization

Weights and bias are updated using **Gradient Descent** to minimize loss.

---

##  Results

The project generates:

* 📉 **Loss Curve** – Shows training convergence
* 📊 **Confusion Matrix** – Evaluates classification performance

---

##  How to Run

### 1. Clone Repository

```
git clone <your-repo-link>
cd Logistic-Regression-Student-Pass-Fail
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Train Model

```
cd src
python train.py
```

### 4. Make Predictions

```
python predict.py
```

### 5. Generate Visualizations

```
python visualize.py
```

## 📦 Requirements

* Python 3.x
* numpy
* pandas
* matplotlib

---

## 🎯 Key Learnings

* Built Logistic Regression **from scratch**
* Understood **classification vs regression**
* Learned **feature normalization**
* Implemented **gradient descent manually**
* Evaluated model using **confusion matrix**

---

##  Author

Bilal Aamir


