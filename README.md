# Iris Species Prediction using K-Nearest Neighbors (KNN)

This project demonstrates how to implement a **K-Nearest Neighbors (KNN)** classifier using the **Iris dataset**, and provides an interactive **Gradio UI** to:
- Upload a CSV dataset and train the model with user-defined `k`
- Input flower measurements and predict the species
- View model accuracy, confusion matrix, and classification report

---

## Features

-  Upload your own `iris.csv` file
-  Choose the value of `k` (number of neighbors)
-  Train and evaluate a KNN model
-  Predict flower species by entering sepal and petal dimensions
-  Simple and intuitive Gradio web interface

---

## Model and Evaluation

- Algorithm: **K-Nearest Neighbors (KNN)**
- Metrics: 
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)

---

## Demo


https://github.com/user-attachments/assets/ba553f95-1e2b-4322-b2da-80152863ef87




---

##  Requirements

Install dependencies:

```bash
pip install pandas scikit-learn gradio
```
## Run the App
```bash
python app.py
```
This will launch a browser window with two tabs:

Train Model: Upload CSV and enter value of k

Predict: Enter flower measurements to get predictions

## Tech Stack
Python 🐍

Scikit-learn

Pandas

Gradio

## Project Structure
```bash
├── app.py               # Main Gradio script
├── iris.csv             # Dataset (can upload your own)
├── requirements.txt     # Optional dependency list
└── README.md
```          
## Future Improvements

Add model comparison (SVM, Decision Trees, etc.)

Visualize decision boundaries

Export trained model


