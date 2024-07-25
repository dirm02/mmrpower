# MMR Power Analysis

This project estimates the statistical power for detecting moderating effects of categorical variables using Moderated Multiple Regression (MMR).

## Project Structure

```
.
├── backend/
│ ├── init.py
│ ├── app.py
│ ├── mmr_power.py
├── frontend/
│ ├── index.html
│ ├── scripts/
│ │ └── script.js
│ ├── styles/
│ └── style.css
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository.

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the backend server:
   ```sh
   python -m backend.app
   ```
4. Open frontend/mmrpower.html in your browser to use the application.

## Usage

1. Specify the number of moderator-based subgroups (k).

2. Specify the desired Type I Error Rate (α).

3. Select whether you wish to conduct the power analysis based on observable or true values.

4. Enter the required values in the table.

5. Click "Calculate Power" to get the power analysis results.

6. Click "Reset" to clear the form.

## Deploy
I deployed this app using Heroku.
* After completing the project, push the project to heroku:
```sh
git push heroku main
```
* Open the deployed app:
```sh
heroku open
```

You can see the web application in [https://mmrpower-cb02959ddde8.herokuapp.com/].
