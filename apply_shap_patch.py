from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")

# Add the SHAP import.
if "import shap\n" not in text:
    marker = "import numpy as np\n"

    if marker not in text:
        raise SystemExit(
            "Could not find the NumPy import line."
        )

    text = text.replace(
        marker,
        marker + "import shap\n",
        1,
    )

# Create a genuine SHAP explainer from the saved
# background profiles and fitted linear SVM.
anchor = 'MODEL_METADATA = ARTIFACT["metadata"]\n'

shap_setup = '''MODEL_METADATA = ARTIFACT["metadata"]

# Genuine SHAP explainer for the trained linear SVM.
MODEL_SHAP_BACKGROUND = MODEL.named_steps[
    "preprocessor"
].transform(
    ARTIFACT["shap_background_profiles"]
)

if hasattr(MODEL_SHAP_BACKGROUND, "toarray"):
    MODEL_SHAP_BACKGROUND = (
        MODEL_SHAP_BACKGROUND.toarray()
    )

MODEL_SHAP_BACKGROUND = np.asarray(
    MODEL_SHAP_BACKGROUND
)


@st.cache_resource
def build_shap_explainer():
    """Create one reusable SHAP explainer for the fitted SVM."""

    return shap.LinearExplainer(
        MODEL.named_steps["classifier"],
        MODEL_SHAP_BACKGROUND,
    )


SHAP_EXPLAINER = build_shap_explainer()
'''

if "SHAP_EXPLAINER = build_shap_explainer()" not in text:
    if anchor not in text:
        raise SystemExit(
            "Could not find the model metadata line."
        )

    text = text.replace(
        anchor,
        shap_setup,
        1,
    )

# Replace the earlier coefficient-based approximation
# with genuine SHAP values.
old_calculation = '''    fitted_classifier = MODEL.named_steps["classifier"]

    transformed_profile = fitted_preprocessor.transform(student_df)

    if hasattr(transformed_profile, "toarray"):
        transformed_profile = transformed_profile.toarray()

    transformed_profile = np.asarray(transformed_profile)[0]
    predicted_class_id = model_ranking["predicted_class_id"]

    class_coefficients = np.asarray(
        fitted_classifier.coef_[predicted_class_id]
    )

    encoded_contributions = (
        transformed_profile - MODEL_BACKGROUND_MEAN
    ) * class_coefficients
'''

new_calculation = '''    transformed_profile = fitted_preprocessor.transform(student_df)

    if hasattr(transformed_profile, "toarray"):
        transformed_profile = transformed_profile.toarray()

    transformed_profile = np.asarray(
        transformed_profile
    )

    predicted_class_id = model_ranking[
        "predicted_class_id"
    ]

    shap_result = SHAP_EXPLAINER(
        transformed_profile
    )

    shap_values = np.asarray(
        shap_result.values
    )

    if shap_values.ndim == 3:
        # Multiclass LinearSVC output:
        # learner x encoded feature x class
        encoded_contributions = shap_values[
            0,
            :,
            predicted_class_id,
        ]

    elif shap_values.ndim == 2:
        encoded_contributions = shap_values[0]

    else:
        return []
'''

if old_calculation in text:
    text = text.replace(
        old_calculation,
        new_calculation,
        1,
    )

elif (
    "shap_result = SHAP_EXPLAINER("
    not in text
):
    raise SystemExit(
        "Could not locate the current explanation calculation."
    )

old_description = '''This calculation uses the fitted SVM coefficients and the average
    transformed training profile. It avoids an extra runtime dependency
    while preserving a genuine feature-level explanation.'''

new_description = '''This calculation uses genuine SHAP values from the fitted SVM and
    groups encoded contributions into the original learner-profile fields.'''

text = text.replace(
    old_description,
    new_description,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "Genuine SHAP explainability was added successfully."
)
