from flask import Flask, request, jsonify
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tfdf.keras.GradientBoostedTreesModel()
model.load("/app/model")  # Path dalam container Cloud Run, pastikan menempatkan model di lokasi ini

primary_labels  = ['Depression', 'Anger', 'Mania', 'Anxiety', 'Somatic Symptoms', 'Suicidal Ideation', 
                   'Psychosis', 'Sleep Problems', 'Memory', 'Repetitive Thoughts and Behaviors', 
                   'Dissociation', 'Personality Functioning', 'Substance Use']

def convert_to_binary(probability, threshold=0.5):
    return 1 if probability >= threshold else 0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Validate features
        train_features = [f"Q{i}" for i in range(1, 24)]
        missing_features = set(train_features) - set(input_data.keys())
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        # Convert to TensorFlow dataset
        def to_tf_dataset(df, label_keys=[]):
            features = dict(df.drop(label_keys, axis=1, errors='ignore'))
            return tf.data.Dataset.from_tensor_slices(features).batch(1)

        input_tf = to_tf_dataset(input_df)
        
        # Model prediction
        prediction = model.predict(input_tf)

        # Convert predictions to binary
        prediction_result = {
            label: [convert_to_binary(p) for p in prediction[label].flatten()]
            for label in primary_labels
        }
        
        return jsonify(prediction_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
