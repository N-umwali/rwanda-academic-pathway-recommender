from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")


# ---------------------------------------------------------
# Add dataclass import
# ---------------------------------------------------------

if "from dataclasses import dataclass" not in text:
    import_anchor = "import shap\n"

    if import_anchor not in text:
        raise SystemExit(
            "Could not find the SHAP import. "
            "No changes were saved."
        )

    text = text.replace(
        import_anchor,
        import_anchor + "from dataclasses import dataclass\n",
        1,
    )


# ---------------------------------------------------------
# Add the StudentProfile domain class
# ---------------------------------------------------------

student_profile_class = '''
@dataclass(frozen=True)
class StudentProfile:
    """
    Structured learner-profile object used by the
    recommendation system.

    The class validates the nine required inputs and
    converts them into the exact dictionary keys expected
    by the trained model and guidance functions.
    """

    education_type: str
    pathway: str
    stream_or_trade: str
    best_subject: str
    weakest_subject: str
    interest_area: str
    average_score_range: str
    digital_skill_level: str
    career_cluster: str

    def __post_init__(self):
        required_values = {
            "Education Type": self.education_type,
            "Pathway": self.pathway,
            "Stream or TVET Trade": self.stream_or_trade,
            "Strongest Subject or Competency": self.best_subject,
            "Subject or Competency Needing Support": self.weakest_subject,
            "Interest Area": self.interest_area,
            "Average Score Range": self.average_score_range,
            "Digital Skill Level": self.digital_skill_level,
            "Career Cluster": self.career_cluster,
        }

        missing_fields = [
            field_name
            for field_name, value in required_values.items()
            if not str(value or "").strip()
        ]

        if missing_fields:
            raise ValueError(
                "Please complete the following learner-profile "
                "fields: "
                + ", ".join(missing_fields)
            )

        allowed_education_types = {
            "General Education",
            "TVET",
        }

        if self.education_type not in allowed_education_types:
            raise ValueError(
                "Education Type must be either "
                "'General Education' or 'TVET'."
            )

    @property
    def is_tvet(self):
        """Return True when the learner follows a TVET route."""

        return self.education_type == "TVET"

    def to_dict(self):
        """
        Convert the object into the feature names expected by
        the model pipeline and recommendation services.
        """

        return {
            "EducationType": self.education_type,
            "Pathway": self.pathway,
            "Stream_or_Trade": self.stream_or_trade,
            "BestSubject": self.best_subject,
            "WeakestSubject": self.weakest_subject,
            "InterestArea": self.interest_area,
            "AverageScoreRange": self.average_score_range,
            "DigitalSkillLevel": self.digital_skill_level,
            "CareerCluster": self.career_cluster,
        }


'''

if "class StudentProfile:" not in text:
    class_anchor = "BASE_DIR = Path(__file__).resolve().parent"

    if class_anchor not in text:
        raise SystemExit(
            "Could not locate BASE_DIR in app.py. "
            "No changes were saved."
        )

    text = text.replace(
        class_anchor,
        student_profile_class + class_anchor,
        1,
    )


# ---------------------------------------------------------
# Replace the plain profile dictionary with the class
# ---------------------------------------------------------

old_profile_block = '''                profile = {
                    "EducationType": education_type,
                    "Pathway": pathway,
                    "Stream_or_Trade": stream_or_trade,
                    "BestSubject": best_subject,
                    "WeakestSubject": weakest_subject,
                    "InterestArea": interest_area,
                    "AverageScoreRange": average_score_range,
                    "DigitalSkillLevel": digital_skill_level,
                    "CareerCluster": career_cluster,
                }

                try:
                    recommended_program, recommended_bridge_course, alternative_pathway, source = recommend_student(profile)
'''

new_profile_block = '''                try:
                    learner_profile = StudentProfile(
                        education_type=education_type,
                        pathway=pathway,
                        stream_or_trade=stream_or_trade,
                        best_subject=best_subject,
                        weakest_subject=weakest_subject,
                        interest_area=interest_area,
                        average_score_range=average_score_range,
                        digital_skill_level=digital_skill_level,
                        career_cluster=career_cluster,
                    )

                    profile = learner_profile.to_dict()

                    recommended_program, recommended_bridge_course, alternative_pathway, source = recommend_student(
                        profile
                    )
'''

if old_profile_block not in text:
    raise SystemExit(
        "Could not find the current learner-profile block. "
        "No changes were saved."
    )

text = text.replace(
    old_profile_block,
    new_profile_block,
    1,
)


path.write_text(
    text,
    encoding="utf-8",
)

print(
    "StudentProfile class added and connected successfully."
)
