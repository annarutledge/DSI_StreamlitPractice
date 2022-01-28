import numpy as np

def predict_flower(trained_classifier, input_features):
    """
    Predict Iris type
    Args: 
        trained_claasifier (scikit-learn estimator): trained classifier
        input_features (np.array like): 4 numeric values
    Returns:
        np.array of string dtype: the model's prediction

    """
    return trained_classifier.predict([input_features])