# Project Summary: ANN_based_Churn_Prediction_Model

This document provides a summary of the files and components within the ANN_based_Churn_Prediction_Model project. The project aims to predict customer churn using an Artificial Neural Network.

## Files and Components:

### 1. `Churn_Modelling.csv`
*   **Purpose:** This CSV file is the dataset used for training and evaluating the customer churn prediction model.
*   **Characteristics:** It contains 10,000 customer records with 14 columns, including various customer attributes (features like CreditScore, Geography, Gender, Age, Tenure, Balance, etc.) and a target variable 'Exited' (indicating whether a customer churned).

### 2. `model.h5`
*   **Purpose:** This file stores the trained Artificial Neural Network (ANN) model.
*   **Format:** It's a Keras/TensorFlow model saved in HDF5 format.
*   **Function:** It contains the learned network architecture (layers, weights, biases) and is loaded by `app.py` to make churn predictions on new customer data.

### 3. Preprocessing Objects (`.pkl` files)
These files are Python objects serialized using `pickle`. They store pre-fitted `scikit-learn` transformers used for data preprocessing in `app.py`.
    *   **`scaler.pkl`**:
        *   Contains a fitted `StandardScaler` for scaling numerical features (e.g., Age, Balance) to have zero mean and unit variance. This ensures consistent transformation of new input data to match the data the model was trained on.
    *   **`label_encoder_gender.pkl`**:
        *   Contains a fitted `LabelEncoder` for converting the categorical 'Gender' feature (e.g., "Male", "Female") into a numerical representation.
    *   **`one_hot_encoder_geo.pkl`**:
        *   Contains a fitted `OneHotEncoder` for converting the categorical 'Geography' feature (e.g., "France", "Germany", "Spain") into a one-hot encoded numerical format.

### 4. `app.py`
*   **Purpose:** This is the main application file that creates a user-facing web interface using Streamlit.
*   **Functionality:**
    *   Loads the pre-trained model (`model.h5`) and preprocessing objects (`.pkl` files).
    *   Provides a UI for users to input customer details (Geography, Gender, Age, Balance, etc.).
    *   Preprocesses the input data using the loaded encoders and potentially a scaler.
    *   Feeds the processed data to the model to get a churn prediction probability.
    *   Displays the result (likely/unlikely to churn) and the probability to the user.

### 5. `requirements.txt`
*   **Purpose:** Lists the Python packages and their versions required to run the project.
*   **Key Libraries:**
    *   `tensorflow`: For the ANN model.
    *   `pandas`: For data manipulation.
    *   `numpy`: For numerical computation.
    *   `scikit-learn`: For data preprocessing.
    *   `tensorboard`: For TensorFlow visualization (likely used during model training).
    *   `matplotlib`: For plotting (likely used in notebooks).
    *   `streamlit`: For building the web application.

### 6. Jupyter Notebooks
    *   **`experiments.ipynb`**:
        *   **Likely Role:** Workspace for model development, including data loading (`Churn_Modelling.csv`), exploratory data analysis, extensive preprocessing, ANN model definition, training, evaluation, and saving of the model (`model.h5`) and preprocessing objects (`.pkl` files).
    *   **`prediction.ipynb`**:
        *   **Likely Role:** Used for loading the trained model and preprocessors to make and analyze predictions. This might be for batch predictions, more detailed analysis of individual predictions, or as a script-based alternative to `app.py`.

### 7. `README.md`
*   **Purpose:** Provides a very brief title for the project.
*   **Content:** States "ANN_based_Churn_Prediction_Model". It currently lacks detailed information about setup, usage, etc.

### 8. `LICENSE`
*   **Purpose:** Specifies the terms and conditions under which the project's software can be used, modified, and distributed. The specific license type (e.g., MIT, GPL) is not determined from this summary alone but its presence signifies that legal terms are defined.
