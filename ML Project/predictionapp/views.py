# views.py
import joblib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

# Load your pre-trained model
model = joblib.load("C:\\Users\\DHINESH KUMAR S\\Downloads\\best_model (1).pkl")

def predict_heart_disease(request):
    if request.method == 'POST':
        # Get input values from the form
        form_data = request.POST

        # Validate and process the form data
        try:
            # Convert form data to float and handle missing or invalid values
            test_input = np.array([[   
                float(form_data.get('HighBP', 0)),  # Default to 0 if value is missing
                float(form_data.get('HighChol', 0)),
                float(form_data.get('CholCheck', 0)),
                float(form_data.get('BMI', 0)),
                float(form_data.get('Smoker', 0)),
                float(form_data.get('Stroke', 0)),
                float(form_data.get('Diabetes', 0)),
                float(form_data.get('PhysActivity', 0)),
                float(form_data.get('Fruits', 0)),
                float(form_data.get('Veggies', 0)),
                float(form_data.get('HvyAlcoholConsump', 0)),
                float(form_data.get('AnyHealthcare', 0)),
                float(form_data.get('NoDocbcCost', 0)),
                float(form_data.get('GenHlth', 0)),
                float(form_data.get('MentHlth', 0)),
                float(form_data.get('PhysHlth', 0)),
                float(form_data.get('DiffWalk', 0)),
                float(form_data.get('Sex', 0)),
                float(form_data.get('Age', 0)),
                float(form_data.get('Education', 0)),
                float(form_data.get('Income', 0))
            ]], dtype=float).reshape(1, -1)  # Ensure correct shape for model input

            # Prediction
            try:
                prediction = model.predict(test_input)

                # Interpret the prediction result
                if prediction[0] == 1:
                    result = 'Heart Disease Detected'
                elif prediction[1] == 0:
                    result = 'No Heart Disease Detected'
                else:
                    result = 'Error: Unexpected prediction output'

            except Exception as e:
                result = f"Error during prediction: {e}"

        except ValueError as e:
            # Catch invalid input and provide error feedback
            result = f"Invalid input: {e}. Please ensure all fields are correctly filled."

        # Render result to template
        return render(request, 'predictionapp/result.html', {'result': result})

    # Render the form page if request is not POST
    return render(request, 'predictionapp/predict.html')
