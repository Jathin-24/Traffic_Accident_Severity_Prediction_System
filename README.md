# 🚦 Real-Time Traffic Accident Severity Prediction System

## 🎯 Objective

Build an end-to-end Machine Learning system that predicts the severity level of road accidents based on environmental, traffic, and weather data. The goal is to assist traffic management and emergency services in proactive response and resource allocation.

---

## 📁 Dataset Description

The dataset contains detailed information about US road accidents, including environmental, traffic, and weather conditions, after preprocessing.

| **Feature**                     | **Description**                                                   |
| :------------------------------ | :---------------------------------------------------------------- |
| `Source`                        | Source of the accident data                                       |
| `Severity`                      | Severity of the accident (target variable: 1-4)                   |
| `Start_Lat`                     | Latitude at the start of the accident                             |
| `Start_Lng`                     | Longitude at the start of the accident                            |
| `Distance(mi)`                  | The length of the road segment affected by the accident           |
| `State`                         | US State where the accident occurred (encoded)                    |
| `Timezone`                      | Timezone of the accident location (encoded)                       |
| `Temperature(F)`                | Temperature at the time of the accident in Fahrenheit             |
| `Wind_Chill(F)`                 | Wind Chill temperature at the time of the accident in Fahrenheit  |
| `Humidity(%)`                   | Humidity at the time of the accident in percentage                |
| `Pressure(in)`                  | Atmospheric pressure at the time of the accident in inches        |
| `Visibility(mi)`                | Visibility at the time of the accident in miles                   |
| `Wind_Direction`                | Cardinal direction of the wind (encoded)                          |
| `Wind_Speed(mph)`               | Wind speed at the time of the accident in miles per hour          |
| `Precipitation(in)`             | Precipitation amount at the time of the accident in inches        |
| `Weather_Condition`             | Description of the weather at the time of the accident (encoded)  |
| `Amenity`                       | Presence of an amenity in the vicinity (Boolean)                  |
| `Bump`                          | Presence of a speed bump (Boolean)                                |
| `Crossing`                      | Presence of a pedestrian crossing (Boolean)                       |
| `Give_Way`                      | Presence of a give way sign (Boolean)                             |
| `Junction`                      | Presence of a road junction (Boolean)                             |
| `No_Exit`                       | Presence of a no exit road segment (Boolean)                      |
| `Railway`                       | Presence of a railway crossing (Boolean)                          |
| `Roundabout`                    | Presence of a roundabout (Boolean)                                |
| `Station`                       | Presence of a station (Boolean)                                   |
| `Stop`                          | Presence of a stop sign (Boolean)                                 |
| `Traffic_Calming`               | Presence of traffic calming measures (Boolean)                    |
| `Traffic_Signal`                | Presence of a traffic signal (Boolean)                            |
| `Sunrise_Sunset`                | Indicates if it's day or night (encoded)                          |
| `Civil_Twilight`                | Indicates civil twilight conditions (encoded)                     |
| `Nautical_Twilight`             | Indicates nautical twilight conditions (encoded)                  |
| `Astronomical_Twilight`         | Indicates astronomical twilight conditions (encoded)              |
| `Duration_in_minutes`           | Duration of the accident event in minutes                         |
| `Start_Hour`                    | Hour of the day when the accident started                         |
| `Start_Weekday`                 | Day of the week when the accident started                         |
| `Start_Month`                   | Month of the year when the accident started                       |
| `Start_Year`                    | Year when the accident started                                    |
| `Weather_Hour`                  | Hour of the day for the weather observation                       |
| `Weather_Weekday`               | Day of the week for the weather observation                       |

Target variable:

- `Severity` — Indicates the severity of the accident, from 1 (low) to 4 (high).

Data Source:

- Kaggle Dataset: [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

## Key Features

- 📌 Data Preprocessing: Handling missing values, feature engineering (time-based features, duration), encoding categorical variables
- 🤖 Model Building: Random Forest Classifier
- 📊 Model Evaluation: Accuracy, Classification Report, Confusion Matrix
- 🧠 Interpretability: Feature Importance
- 🌐 Web App: Interactive Streamlit dashboard for real-time and batch predictions

---

## 🧰 Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (RandomForestClassifier, LabelEncoder)
- Joblib (for model serialization)
- Streamlit (for dashboard)

---

## 📊 Feature Importance & Model Performance Summary

**Top features influencing accident severity (from Random Forest Model):**

- `Start_Lng`
- `Start_Lat`
- `Duration_in_minutes`
- `Source`
- `Distance(mi)`
- `Start_Year`
- `Pressure(in)`
- `Temperature(F)`
- `Humidity(%)`
- `State`

## Model Performance Summary

| Model           | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------|----------|-----------|--------|----------|---------|
| Random Forest   | 0.8640   | 0.85      | 0.86   | 0.85     | N/A     |

---

### 📉 Detailed Model Results

#### Random Forest Classifier

**Accuracy**: 0.8640

**Classification Report**:
```
              precision    recall  f1-score   support

           1       0.74      0.54      0.63      1361
           2       0.89      0.95      0.92    123050
           3       0.71      0.57      0.64     26051
           4       0.63      0.17      0.27      4106

    accuracy                           0.86    154568
   macro avg       0.74      0.56      0.61    154568
weighted avg       0.85      0.86      0.85    154568
```

**Confusion Matrix**:
```
[[   741    542     78      0]
 [   171 117156   5466    257]
 [    75  10873  14951    152]
 [     9   2963    437    697]]
```
#### Gradient Boosting

**Accuracy**: 0.8274

**Classification Report**:
```
              precision    recall  f1-score   support

           1       0.65      0.25      0.36      1361
           2       0.84      0.96      0.90    123050
           3       0.66      0.34      0.45     26051
           4       0.60      0.05      0.08      4106

    accuracy                           0.83    154568
   macro avg       0.69      0.40      0.45    154568
weighted avg       0.81      0.83      0.80    154568
```

**Confusion Matrix**:
```
[[   345    985     31      0]
 [   112 118526   4351     61]
 [    66  17091   8831     63]
 [     8   3726    185    187]]
```

#### XG Boost 

**Accuracy**: 0.8626

**Classification Report**:
```
              precision    recall  f1-score   support

           0       0.68      0.55      0.61      1361
           1       0.89      0.95      0.92    123050
           2       0.71      0.58      0.64     26051
           3       0.63      0.17      0.27      4106

    accuracy                           0.86    154568
   macro avg       0.73      0.56      0.61    154568
weighted avg       0.85      0.86      0.85    154568
```

**Confusion Matrix**:
```
[[   746    554     60      1]
 [   238 116700   5885    227]
 [    98  10592  15173    188]
 [    12   2985    401    708]]
```

---
## 🔧 Prerequisites

Before you begin, make sure you have the following tools installed on your system.

---

## Step 1: Install Prerequisites

#### 1.1 Install Git

- Download: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Install Git with default options.
- Confirm:

```bash
git --version
```

#### 1.2 Install Python (3.10+)

- Download: [https://www.python.org/downloads](https://www.python.org/downloads)
- IMPORTANT: Check ✅ "Add Python to PATH" during install.
- Confirm:

```bash
python --version
pip --version
```

#### 1.3 Install Visual Studio Code

- Download: [https://code.visualstudio.com](https://code.visualstudio.com)
- Install and open VS Code.

---

## Step 2: Clone the Repository

```bash
git clone https://github.com/Jathin-24/Traffic_Accident_Severity_Prediction_System.git
cd Traffic_Accident_Severity_Prediction_System
```

---

## Step 3: Create Python Virtual Environment

#### Using `venv`:

```bash
python -m venv environ
environ\Scripts\activate
```

> You should now see `(environ)` in your terminal prompt.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Open Project in VS Code

```bash
code .
```

If `code` command doesn't work:

1. Open VS Code manually.
2. Go to `File > Open Folder`.
3. Select the cloned `Traffic_Accident_Severity_Prediction_System` folder.

---
# NOTE
We are considering that the user is not having enough computation power for processing 7.7 million rows in the dataset so we will use google colab for the model development

## Step 6: Download the Dataset

This project uses the **US Accidents Dataset** from Kaggle.

### 6.1 Download the Dataset

1. Go to the dataset page:
   [https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Download the ZIP file.
3. Extract it to obtain the CSV file, such as:

 *  `US_Accidents_March23.csv`

---

## Step 7: Upload the Dataset to Google Drive

To ensure the notebooks run smoothly in Google Colab, upload the dataset to Google Drive.

### 7.1 Create Dataset Folder

1. Open Google Drive: [https://drive.google.com](https://drive.google.com)
2. Navigate to **MyDrive**
3. Create a folder named:

```
dataset
```

### 7.2 Upload CSV File

Upload your extracted dataset (CSV file) into:

```
MyDrive/dataset/
```

Your directory should look like:

```
MyDrive/dataset/US_Accidents_*.csv
```

---

## Step 8: Open the Notebooks in Google Colab

### 8.1 Notebook Files

This project provides several Jupyter notebooks:

* `Random_Forest_Complete (1).ipynb`
* `Gradient_Boosting_Complete.ipynb`
* `XG_Boost_Complete.ipynb`


### 8.2 Open in Google Colab

**Option A (Recommended)**

1. Upload the notebooks to Google Drive.
2. Right-click on the notebook.
3. Select:

   ```
   Open with → Google Colab
   ```

**Option B (If using GitHub)**

1. Open the notebook in GitHub.
2. Replace the URL prefix:

   * Change `github.com` → `colab.research.google.com/github`
3. Colab will open the notebook directly.

---

## Step 9: Mount Google Drive in Colab

Before loading the dataset, mount Google Drive inside your notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

After mounting, your dataset will be accessible at:

```
/content/drive/MyDrive/dataset/US_Accidents_*.csv
```

Use this path when loading the data:

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/dataset/US_Accidents_March23.csv', nrows = 1000)
df.head()
```

---

## Step 10: Run the Notebooks

### 1. `Random_Forest_Complete (1).ipynb`

* Preprocessing
* Feature Engineering
* Sampling of data
* Train Random Forest
* Saves and downloads the Random Forest Model to the local system

### 2. `Gradient_Boosting_Complete.ipynb`

* Preprocessing
* Feature Engineering
* Sampling of data
* Train Gradient Boosting model
* Saves and downloads the Gradient Boosting model to the local system

### 3. `XG_Boost_Complete.ipynb`

* Preprocessing
* Feature Engineering
* Sampling of data
* Train XG Boost model
* Saves and downloads the XG Boost model to the local system

---

## Step 11: Move the Downloaded Model to the cloned Project Repository in VS Code

After downloading the trained model file (e.g., `random_forest_model.pkl`) from Google Colab:

1. Locate the cloned project repository on your system.
2. Create a directory named:

```
models
```

3. Move the downloaded model file into:

```
Traffic_Accident_Severity_Prediction_System/models/
```

---

## Step 12: Run the Streamlit Application

Once the trained model has been moved into the `models/` directory, you can run the Streamlit application locally.

### 12.1 Activate Your Virtual Environment

If you created a virtual environment (recommended), activate it:

**Windows (PowerShell or CMD):**

```
venv\Scripts\activate
```

**macOS / Linux:**

```
source venv/bin/activate
```

---

### 12.2 Install Dependencies

Ensure all required Python packages are installed:

```
pip install -r requirements.txt
```

---

### 12.3 Run the Streamlit App

Execute the application using Streamlit:

```
streamlit run app7.py
```
This will:

* Launch the web application locally
* Load the trained model from the `models/` folder
* Allow you to perform inference using the deployed UI

Once the app starts, Streamlit will open automatically in your browser (or will display a local URL you can visit manually).

---

## 13. Prediction using various models

If you want to load a specific model for prediction replace the model name in app7.py with specific model file name (*.pkl)

---

## 🚀 How to Use the Accident Severity Prediction System ?

Once the Streamlit application is running and accessible via the provided URL:

1.  **Single Prediction:**
    *   Navigate to the "📝 Single Prediction" tab.
    *   You will see a form with various input fields for location, time, weather, and road features.
    *   Enter the relevant values for a hypothetical (or real-world) accident scenario.
    *   Click the "🔮 Predict Severity" button.
    *   The system will display the predicted accident severity level (1-4) along with confidence scores and an interpretation.

2.  **Batch Prediction from CSV:**
    *   Navigate to the "📊 Batch Prediction" tab.
    *   You can download a sample CSV template to understand the required input format.
    *   Prepare your own CSV file with multiple accident records, ensuring it matches the template's column structure.
    *   Upload your CSV file using the "Upload CSV file" button.
    *   Click "🔮 Predict All" to get predictions for all records in your uploaded file.
    *   The results, including predicted severity and probabilities, will be displayed in a table and can be downloaded as a new CSV file.

---

## 📝 Final Notes

- ✅ Ensure the model file (`random_forest_model.pkl`) is present in your Colab environment for the Streamlit app to function correctly.
- 🔒 No personal data is used in this system; all input is based on accident features.

---

## 🧾 License

This project is for educational and demonstrative use, focusing on the application of machine learning in traffic management.

## 🙏 Acknowledgments

- Internship Host: Flipkart Pvt Ltd
- Project Mentor: Sana Srinadh (Data Science & Machine Learning Domain)
- Dataset: [US Accidents from Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

## Contact

For questions or further information, please refer to the project's Colab notebook.
