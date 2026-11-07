import json
import pickle

# 1. Defining the lead data exactly as provided
lead_data = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# 2. Loading the pre-trained pipeline model artifact
model_file = "pipeline.bin"
try:
    with open(model_file, "rb") as f_in:
        pipeline = pickle.load(f_in)

    # 3. Predicting probability
    # Usually, a scikit-learn pipeline expects an array/list of dictionaries
    # or a pandas DataFrame, depending on how your preprocessor was built.
    prediction = pipeline.predict_proba([lead_data])[0]

    # 4. Extracting conversion probability (typically class 1, index 1)
    # prediction[0] is the probability of non-conversion (Class 0)
    # prediction[1] is the probability of conversion (Class 1)
    conversion_prob = prediction[1]

    # 5. Outputing the result rounded to three decimal places
    print(f"The probability of conversion is: {conversion_prob:.3f}")

except FileNotFoundError:
    print(
        f"Error: '{model_file}' not found. Please ensure the file is in your current working directory."
    )
except Exception as e:
    print(f"An error occurred during prediction: {e}")