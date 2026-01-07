# Satellite-Based Property Price Prediction (Multi-Modal Learning)

This project predicts **property prices** by combining **tabular property data** with **satellite imagery** using a **multi-modal deep learning approach**.  
A **pre-trained Inception-V3 CNN** extracts spatial features from satellite images, which are fused with tabular features using a **neural network–based feature fusion model**.

---

##  Project Highlights
- Multi-modal learning using **tabular + satellite image data**
- **Inception-V3 (frozen)** for image feature extraction
- **Feature-level (F+I) fusion** using neural networks
- Baseline vs fusion model comparison
- Model interpretability using **Grad-CAM**

---

##  Project Structure

├── EDA_Satellite_Property_Valuation.ipynb
├── preprocessing.ipynb
├── model_training.ipynb # Final fusion model
├── 23112116_report.pdf
├── 23112116_final.csv
├── data_fatecher.py
├── requirements.txt
└── README.md


## ⚙️ Requirements

- Python 3.8 or higher

### Python Libraries
- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- torch  
- torchvision  
- xgboost  
- opencv-python  
- tqdm  

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/Vishalmeena0658/satellite-property-valuation.git
cd satellite-property-valuation

2️⃣ Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate         

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Run the Project
Step 1: Exploratory Data Analysis (EDA)
EDA_Satellite_Property_Valuation.ipynb
Data quality checks
Correlation analysis
Spatial visualization
Satellite image exploration

Step 2: Data Preprocessing
preprocessing.ipynb
Log transformation of target variable
Removal of non-informative columns (ID,Date , ZIP code)
Feature selection
Data preparation for modeling

Step 3: Baseline Model (Tabular Only)
model_training.ipynb
Trains XGBoost regression
Uses only tabular property features
Serves as baseline for comparison

Step 4: Final Fusion Model (Main Model)
Satellite_Property_Valuation.ipynb
Extracts 2048-d image embeddings using Inception-V3
Generates 40-d tabular embeddings
Applies feature-level fusion
Trains neural network (512 → 64 → 1)
Evaluates RMSE and R²
Generates Grad-CAM visualizations

🧠 Model Architecture
Satellite Image → Inception-V3 → 2048-d Image Embedding
Tabular Features → Dense Layer → 40-d Tabular Embedding
→ Feature Concatenation (F+I Fusion)
→ FC (512 → 64 → 1)
→ Predicted Property Price

📊 Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

Baseline and fusion models are compared to analyze the impact of satellite imagery.

🔍 Model Interpretability
Grad-CAM visualizations highlight spatial regions influencing predictions
Shows differences between low-price and high-price properties
Improves trust and explainability of the fusion model
