# OpacGuard – German Credit Risk Project

OpacGuard is a Machine Learning + Web App project that predicts **credit risk** using the German Credit dataset.  
The project consists of:

- **Backend (Flask + Python)** → model training, serving predictions
- **Frontend (React)** → user interface for input & visualization

---

## 🚀 Features
- ML model trained on German Credit dataset
- REST API using Flask
- React-based frontend with live requests to backend
- CORS enabled for smooth local development
- Modular code structure (`backend/`, `frontend/`)

---

## 🗂 Project Structure
Opac_new/
│── backend/          # Flask API, ML model, requirements.txt
│── frontend/         # React frontend (npm/yarn)
│── venv/             # Local Python virtual environment (ignored by Git)
│── requirements.txt  # Python dependencies
│── .gitignore        # Ignore rules for backend + frontend
│── german.data       # Dataset files (if used locally)
---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/codemidas04/Opac_new.git
cd Opac_new

2. Backend (Flask API)

Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Install Dependencies
pip install -r requirements.txt

Run Backend
cd backend
python app.py
👉 Flask will run on: http://localhost:5000


3. Frontend (React)

Go to Frontend Folder
cd frontend

Install Dependencies
npm install

Start Frontend
npm start

👉 React will run on: http://localhost:3000


🔄 Workflow for Team Members

	•	Pull latest changes:
    git pull origin main

	•	Add new Python packages:
pip install package-name
pip freeze > requirements.txt
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
	•	Always install dependencies using requirements.txt (backend) and npm install (frontend).
	•	For dataset files, use data/ locally — don’t push large datasets to GitHub.

👥 Contributors
	•	Aditya
	•	Chaitanya
	•	Chirag

---

⚡ Next step:  
Run these commands to add it to GitHub:

```bash
cd ~/Downloads/Opac_new
touch README.md
open -e README.md   # paste the above block
git add README.md
git commit -m "Add project README with setup instructions"
git push

