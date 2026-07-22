# Rwanda Academic Pathway and Bridge Course Recommendation System

## Project Overview

The Rwanda Academic Pathway and Bridge Course Recommendation System is a Streamlit-based decision-support prototype developed for Rwandan General Education and TVET learners. It uses a trained Support Vector Machine model to rank broad academic fields, then applies pathway-aware guidance to refine the result into a specific program direction, bridge-course preparation, and an alternative pathway.

The system is advisory. It does not replace official university admission requirements, institutional guidance, or professional academic counselling.

## Main Functions

- Collects a structured learner profile for General Education and TVET routes.
- Predicts one of 16 broad academic program categories using a trained LinearSVC pipeline.
- Refines the broad prediction into a learner-facing program recommendation.
- Keeps TVET recommendations grounded in the learner's current trade while recording broader interests as possible transition goals.
- Maps the recommendation to bridge courses and an alternative pathway.
- Uses SHAP values from the fitted linear model to identify the profile features that most influenced the broad-field prediction.
- Generates a downloadable guidance report.
- Collects prototype feedback on usefulness, clarity, and usability.

## Recommendation Architecture

```text
Learner profile
      ↓
Input validation and model-vocabulary preparation
      ↓
LinearSVC ranks 16 broad academic fields
      ↓
Pathway and TVET-alignment refinement
      ↓
Specific program recommendation
      ↓
Bridge-course and alternative-pathway mapping
      ↓
SHAP-supported learner-friendly explanation
      ↓
Human review and verification of official requirements
```

The machine-learning prediction is produced first. The guidance layer then converts the broad model output into a practical recommendation and applies pathway-specific safeguards. SHAP explains the trained model's broad-category decision; it does not claim that the final specific program was directly predicted by the model.

## Object-Oriented Components

The application uses the following main classes:

- `StudentProfile`: validates and structures the nine learner inputs.
- `RecommendationResult`: stores the program, bridge courses, alternative pathway, source, and predicted broad category.
- `AcademicPathwayRecommender`: coordinates model ranking, program refinement, bridge-course mapping, and alternative-pathway selection.
- `RecommendationExplainer`: converts SHAP-supported model influences into learner-friendly guidance.

The final deployment is kept in one Streamlit entry file for deployment simplicity. Further modular separation into dedicated domain, policy, and UI packages is identified as future maintainability work.

## Model and Evaluation

- Model: Support Vector Machine (`LinearSVC`)
- Target categories: 16 broad academic fields
- Training records: 8,484
- Test records: 2,121
- Random state: 42

| Metric | Proposal Target | Final Result | Outcome |
|---|---:|---:|---|
| Accuracy | at least 80% | 78.31% | Slightly below target |
| Macro F1 | at least 75% | 62.85% | Below target |
| Micro F1 | Not separately specified | 78.31% | Reported for completeness |
| Bridge-course relevance | at least 80% | Not yet formally measured | Requires user/advisor evaluation |
| Explanation clarity | at least 70% | Not yet formally measured | Requires feedback evidence |
| Usability | at least 70% | Not yet formally measured | Requires feedback evidence |

The model performs well enough to demonstrate the complete recommendation workflow, but the class imbalance and variation across 16 categories reduce Macro F1. The system should therefore be treated as a prototype and used with human review.

Evaluation evidence is stored in `report_outputs/`, including:

- raw and normalized confusion matrices;
- final test metrics;
- model comparison;
- subgroup performance;
- global SHAP importance;
- final model metadata and results.

## Project Structure

```text
rwanda-academic-pathway-recommender/
├── .streamlit/
│   └── config.toml
├── .gitattributes
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── models/
│   └── academic_pathway_model_v2.joblib
├── notebook/
│   └── Explainable_Academic_Pathway_Recommender_Model.ipynb
├── data/
│   ├── Academic Pathway and Bridge Course Recommendation Survey for Rwandan Students (Responses).xlsx
│   └── rwandan_student_pathway_dataset.xlsx
├── report_outputs/
├── screenshots/
└── static/
    ├── branding/
    ├── icons/
    └── images/
```

## Technologies

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn 1.6.1
- SHAP 0.50.0
- Joblib
- OpenPyXL
- Git and Git LFS

## Installation

```bash
git clone https://github.com/N-umwali/rwanda-academic-pathway-recommender.git
cd rwanda-academic-pathway-recommender
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Model Artifact

The deployed model bundle is stored at:

```text
models/academic_pathway_model_v2.joblib
```

The bundle includes the fitted model pipeline, label encoder, feature order, bridge-course mapping, ordinal category orders, SHAP background profiles, and model metadata.

## Testing Evidence

The application was tested using:

- General Education profiles from Mathematics and Sciences, Arts and Humanities, and Languages;
- TVET profiles from different technical trades;
- aligned and deliberately unrelated interest/career combinations;
- different score ranges and digital-skill levels;
- local model loading, recommendation generation, SHAP explanation, and report download;
- model save-and-reload verification in the notebook.

Screenshots are stored in `screenshots/`. Before final submission, the folder should include at least one screenshot of the deployed Streamlit Cloud application showing the live URL and one complete TVET recommendation output.

## Data, Governance, and Responsible Use

The application does not request national ID, phone number, exact address, or other unnecessary personal identifiers. Learner inputs are used for the current recommendation session. Feedback is collected only for prototype evaluation.

Recommendations must be checked against official admission requirements and reviewed with academic advisors, schools, guardians, or relevant institutions. The system is not an official MINEDUC, REB, RTB, or university admission platform.

## Limitations

- The training dataset does not represent every learner and institution in Rwanda.
- Final accuracy and Macro F1 are below the proposal targets.
- Some target categories have fewer examples than others.
- The specific-program refinement and bridge-course mappings require future validation against official institutional requirements.
- TVET continuation guidance relies partly on predefined trade-to-field mappings.
- User-centred success measures require formal student and advisor feedback.
- The deployment code remains in one large Streamlit file and should be separated into modules in a later version.

## Future Work

- Collect more representative, real learner and advisor data.
- Validate program eligibility and bridge courses with institutions and education authorities.
- Improve minority-class performance and recalibrate the target categories.
- Add automated recommendation-integrity tests.
- Calculate dashboard ratings from real feedback rather than fixed prototype values.
- Separate model services, eligibility policy, reporting, and UI components into dedicated modules.
- Add school-, district-, and national-level analytics after governance approval.

## Deployment

Streamlit Cloud application:

```text
https://rwanda-academic-pathway-recommender.streamlit.app/
```

After every final GitHub update, verify that the deployment loads the model, generates both General Education and TVET recommendations, displays SHAP-supported explanations, and downloads the guidance report.

## Demo Video

```text
https://youtu.be/2N_InuK50Po
```

## Author

**Umwali Noella**  
African Leadership University  
Machine Learning Specialization  
Final Capstone Project
