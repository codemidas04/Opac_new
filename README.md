# OpacGuard – German Credit Risk Project

OpacGuard is a Machine Learning + Web App project that predicts **credit risk** using the German Credit dataset.  
The project consists of:

- **Backend (Flask + Python)** → model training, serving predictions
- **Frontend (React)** → user interface for input & visualization
- **Streamlit App** → alternative interface for exploration
- **Explainability (SHAP)** → interpretable ML predictions

---

## 🚀 Features
- ML model trained on German Credit dataset
- REST API using Flask
- React-based frontend with live requests to backend
- Streamlit interface for rapid prototyping
- Explainability with **SHAP** (SHapley Additive exPlanations)  
  - Visualizes how each feature influences predictions
  - Helps users and stakeholders trust the model
- Modular code structure (`backend/`, `frontend/`, `streamlit_app.py`)

---

## 🗂 Project Structure

Opac_new/
│── backend/          # Flask API, ML model
│── frontend/         # React frontend (npm/yarn)
│── streamlit_app.py  # Streamlit dashboard
│── test.ipynb        # EDA notebook
│── venv/             # Local Python virtual environment (ignored in Git)
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
│── .gitignore        # Ignore rules for backend + frontend
│── german.data       # Dataset files (if used locally)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/codemidas04/Opac_new.git
cd Opac_new

2. Backend (Flask API)
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Install Dependencies
pip install -r requirements.txt

Run Backend
cd backend
python app.py
👉 Flask runs on: http://localhost:5000

3. Frontend (React)

Go to Frontend Folder
cd frontend

Install Dependencies
npm install

Start Frontend
npm start
👉 React runs on: http://localhost:3000

4. Streamlit App

To run the Streamlit dashboard:
streamlit run streamlit_app.py
👉 Streamlit runs on: http://localhost:8501

5. Exploratory Data Analysis (EDA)
jupyter notebook test.ipynb


🔄 Workflow for Team Members
•	Pull latest changes:

git pull origin main

•	Add new Python packages:
pip install package-name
pip freeze | grep package-name >> requirements.txt
git add requirements.txt
git commit -m "Add new dependency"
git push

•	Add new Node packages:
cd frontend
npm install package-name
git add package.json package-lock.json
git commit -m "Add new frontend dependency"
git push

📝 Notes
	•	venv/ and node_modules/ are not tracked in Git (see .gitignore).
	•	Always install dependencies using requirements.txt (at project root) and npm install (frontend).
	•	Use data/ folder locally for datasets — do not push large datasets to GitHub.
	•	For model explainability, we use SHAP to interpret feature contributions.


👥 Contributors
	•	Aditya
	•	Chaitanya
	•	Chirag
segsdgfh


