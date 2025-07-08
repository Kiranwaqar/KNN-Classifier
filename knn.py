import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Global variables for model reuse
knn_model = None
X_test = y_test = None

def train_knn(file, k):
    global knn_model, X_test, y_test
    
    # Load data
    df = pd.read_csv(file)
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Evaluate
    y_pred = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    summary = f"Model trained with k = {k}\n\nAccuracy: {acc:.2f}\n\nConfusion Matrix:\n{cm}\n\nClassification Report:\n\n{report_df.to_string()}"

    return summary

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    if knn_model is None:
        return "Please train the model first by uploading the dataset."
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = knn_model.predict(sample)[0]
    return f"Predicted Species: {prediction}"

# Gradio Interface
upload_interface = gr.Interface(
    fn=train_knn,
    inputs=[
        gr.File(label="Upload Iris CSV"),
        gr.Number(value=3, label="Enter value of k")
    ],
    outputs="text",
    title="Train KNN Classifier on Iris Dataset",
    description="Upload the Iris dataset and train the model with your chosen value of K."
)

predict_interface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text",
    title="Predict Iris Species",
    description="Enter measurements to classify a flower species using the trained KNN model."
)

# Combine interfaces in tabs
gr.TabbedInterface([upload_interface, predict_interface], ["Train Model", "Predict"]).launch()
