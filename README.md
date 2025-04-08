# 🔄 E-Commerce Recommendation System

A recommendation engine for e-commerce event data, using a combination of **collaborative filtering** and **matrix factorization** techniques. Built with both **PySpark (ALS)** and **Surprise (KNN, SVD, NMF)** libraries.

> 💡 After preprocessing, you can train a model, test performance, and receive product recommendations for both existing and new users.

---

## ⚙️ Tech Stack

| Component              | Library / Tool          |
|------------------------|--------------------------|
| Collaborative Filtering| Surprise (`KNN`, `SVD`, `NMF`) |
| Matrix Factorization   | PySpark (`ALS`)          |
| Data Manipulation      | Pandas, PySpark          |
| Evaluation Metrics     | MAE, RMSE                |
| CLI Interface          | Python (interactive)     |

---

## 📁 Project Structure

```bash
├── Data/                   # Raw event CSVs (input)
│   └── events.csv
├── Preprocessed Data/      # Cached CSV after preprocessing
│   └── pd_events.csv
├── recommender.py          # Main model training and CLI code
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
```

## 🚀 Installation

### 🔗 Prerequisites

- Python 3.8+
- Java installed and added to PATH (required for PySpark)
- Pip + virtual environment (recommended)

### 🧰 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ecommerce-recommender.git
cd ecommerce-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```
## 📊 Running the App

### ✅ Step-by-step

```bash
# Run the CLI
python recommender.py
```
### 🧠 Options Available

After launching the app, you can:

#### 🛠️ Train one of the models:
- **ALS** – Apache Spark’s Alternating Least Squares algorithm
- **KNN-item** – Item-based collaborative filtering using cosine similarity
- **KNN-user** – User-based collaborative filtering using cosine similarity
- **SVD** – Singular Value Decomposition for matrix factorization
- **NMF** – Non-negative Matrix Factorization

#### 👤 Choose a recommendation type:
- **Random user** – Generate recommendations for a randomly selected existing user
- **Fixed user** – Use predefined user ID `162285` for consistent output
- **New user** – Recommend based on item popularity and a serendipity score (popularity + randomness)

> ⚠️ **Note**: KNN training is memory intensive. Avoid running multiple KNN models back-to-back without restarting the script to prevent memory issues.

## 🧪 Model Performance Metrics

After training a model, the following evaluation metrics will be printed to assess performance:

- **RMSE** (Root Mean Squared Error) – Measures the average magnitude of prediction error.
- **MAE** (Mean Absolute Error) – Represents the average absolute difference between predictions and actual ratings.

These metrics help compare the accuracy of different recommendation algorithms.

---

## 🔮 Future Improvements

Planned enhancements to improve performance, usability, and interactivity:

- **Add content-based filtering** using item metadata (e.g., categories, brand).
- **Build a web interface** using Streamlit or Flask for easier interaction.
- **Add visualizations** for user-item interactions and recommendation results.
- **Enable export to CSV** for saving personalized recommendation lists.

## 👥 Contributors

- **Julisa Delfin** – Graduate Student, DePaul University
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/julisadelfin/)  
