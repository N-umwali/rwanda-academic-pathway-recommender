# Rwanda Academic Pathway and Bridge Course Recommendation System

## Project Overview

The **Rwanda Academic Pathway and Bridge Course Recommendation System** is a Streamlit-based machine learning application designed to support Rwandan students in making informed academic progression decisions.

The system recommends a suitable **academic program category** and **bridge course** based on a student’s academic profile, education type, pathway, stream or TVET trade, strongest subject, weakest subject, interest area, average score range, digital skill level, and career cluster.

This project was developed as a final capstone product for the Machine Learning specialization at African Leadership University.

---

## Core Functionalities

The application provides the following main functionalities:

* Student academic profile input
* Academic program category recommendation
* Bridge course recommendation
* Alternative pathway guidance
* Explainable recommendation output
* National-level dashboard interface
* Testing with different student profiles and edge cases
* Feedback collection for future improvement

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Git LFS
* GitHub

---

## Project Structure

```text
rwanda-academic-pathway-recommender/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── academic_pathway_model.joblib
│
├── data/
│   └── sample_test_cases.csv
│
├── notebooks/
│   └── Explainable_Academic_Pathway_Recommender_Model.ipynb
│__Deployed app linK/
|     |__https://rwanda-academic-pathway-recommender.streamlit.app/
├── screenshots/
│   ├── 01_home_page.png
│   ├── 02_recommendation_form.png
│   ├── 03_general_education_result.png
│   ├── 04_tvet_result.png
│   ├── 05_edge_case_result.png
│   ├── 06_feedback_page.png
│   └── 07_app_running_terminal.png
│
└── demo/
    └── video_link.txt

```

---

## Related Project Files

| File/Folder        | Description                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `app.py`           | Main Streamlit application file                                                                      |
| `models/`          | Contains the trained machine learning model saved using Joblib                                       |
| `notebooks/`       | Contains the final notebook used for data preparation, model training, testing, and saving the model |
| `data/`            | Contains sample test cases used for demo and testing                                                 |
| `screenshots/`     | Contains screenshots showing product testing and functionality                                       |
| `demo/`            | Contains the link to the 5-minute demo video                                                         |
| `requirements.txt` | Lists the Python libraries required to run the app                                                   |

---

## Installation and Running the App Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/N-umwali/rwanda-academic-pathway-recommender.git
```

### Step 2: Move Into the Project Folder

```bash
cd rwanda-academic-pathway-recommender
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

### Step 5: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, the app will open in the browser.

---

## Model Information

The recommendation model is saved in the following location:

```text
models/academic_pathway_model.joblib
```

The model was trained using student academic profile features such as:

* Education type
* Pathway
* Stream or TVET trade
* Best subject
* Weakest subject
* Interest area
* Average score range
* Digital skill level
* Career cluster

The model output is combined with rule-based recommendation logic to provide more explainable and context-aware academic pathway recommendations.

---

## Testing Results and Strategies

The product was tested using multiple testing strategies to verify that the system works with different student profiles and input values.

### Testing Strategy 1: General Education Student Test

A student from General Education was tested using different pathway, subject, interest, and career cluster values. The system successfully generated a relevant academic program category and bridge course recommendation.

### Testing Strategy 2: TVET Student Test

A TVET learner profile was tested to confirm that the system can provide recommendations for technical and vocational education learners. The system produced a recommendation aligned with the learner’s trade, interest area, and career cluster.

### Testing Strategy 3: Different Data Values Test

The application was tested with different combinations of input values, including different education types, pathways, subjects, digital skill levels, score ranges, and career interests. This helped confirm that the system responds dynamically to different student profiles.

### Testing Strategy 4: Edge Case Test

The system was tested with lower score ranges and less direct academic profiles to check whether it still provides useful bridge course guidance. The system was able to return a recommendation and support course suggestion instead of failing.

### Testing Strategy 5: Local Environment Test

The application was tested locally using Git Bash and Streamlit to confirm that it can run successfully after installing the required dependencies.

Testing screenshots are available in the `screenshots/` folder.

---

## Analysis of Results

The results show that the application achieved the main objectives of the project proposal. The system is able to collect student academic profile information, process the input values, and generate academic pathway recommendations with bridge course suggestions.

The product aligns with the project scope because it focuses on supporting Rwandan students in academic decision-making, especially when transitioning from secondary education or TVET into higher education opportunities.

The system also improves explainability by not only giving a recommendation but also showing why the recommendation was made. This is important because students, parents, academic advisors, and education stakeholders need recommendations that are understandable and not only model-generated.

One limitation is that the current version depends on the available training dataset and predefined recommendation logic. Future versions can improve the model by adding more real student data, advisor feedback, and official program eligibility requirements.

---

## Discussion

The main milestone of this project was building a working recommendation system that connects machine learning with a practical education problem in Rwanda. The project demonstrates how student profile data can be used to support academic guidance and bridge course planning.

The impact of the product is that it can help students receive more personalized academic guidance instead of relying only on general advice. It can also support schools, advisors, and education stakeholders by providing a structured way to recommend academic pathways and readiness support.

The bridge course recommendation is an important part of the system because it does not only tell students where they may fit, but also suggests what they can strengthen before progressing to higher education or career training.

---

## Recommendations and Future Work

Future improvements may include:

* Adding more real student data to improve model accuracy
* Including official university admission requirements
* Adding a downloadable student recommendation report
* Creating separate dashboards for students, advisors, and administrators
* Adding school-level and district-level analytics
* Improving deployment for wider national-level use
* Adding advisor feedback to continuously improve the recommendation logic
* Expanding the system to include more academic programs and TVET pathways

The community can use this product as a starting point for improving career guidance, academic advising, and bridge course planning for students in Rwanda.

---

## Deployment

Deployed app link:

```text
Paste your deployed Streamlit app link here
```

---

## Demo Video

A 5-minute demo video showing the core functionalities of the app is available here:

```text
Paste your demo video link here
```

The video focuses on the main recommendation functionality, different student profile tests, bridge course output, and how the product aligns with the project proposal.

---

## Author

**Umwali Noella**
African Leadership University
Machine Learning Specialization
Final Capstone Project

---

## Repository

GitHub Repository:

```text
https://github.com/N-umwali/rwanda-academic-pathway-recommender
```
