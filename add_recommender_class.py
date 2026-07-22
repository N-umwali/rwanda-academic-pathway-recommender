from pathlib import Path
import re

path = Path("app.py")
text = path.read_text(encoding="utf-8")

new_recommender_section = '''@dataclass(frozen=True)
class RecommendationResult:
    """
    Structured recommendation output returned by the
    academic pathway recommender.
    """

    program: str
    bridge_courses: str
    alternative_pathway: str
    source: str
    predicted_broad_category: str

    def as_tuple(self):
        """Maintain compatibility with the existing UI code."""

        return (
            self.program,
            self.bridge_courses,
            self.alternative_pathway,
            self.source,
        )


class AcademicPathwayRecommender:
    """
    Coordinate model ranking, program refinement,
    bridge-course mapping, and alternative pathways.

    This service keeps the recommendation workflow
    separate from the Streamlit interface.
    """

    def __init__(
        self,
        bridge_course_map=None,
        broad_bridge_course_map=None,
    ):
        self.bridge_course_map = (
            bridge_course_map
            or PROGRAM_CATEGORY_TO_BRIDGE_COURSE
        )

        self.broad_bridge_course_map = (
            broad_bridge_course_map
            or MODEL_BROAD_CATEGORY_TO_BRIDGE_COURSE
        )

        self.source_label = (
            "Learner profile and academic pathway guidance"
        )

    @staticmethod
    def _profile_dict(student_profile):
        """
        Accept either a StudentProfile object or an existing
        dictionary without changing the model input format.
        """

        if isinstance(student_profile, StudentProfile):
            return student_profile.to_dict()

        if isinstance(student_profile, dict):
            return dict(student_profile)

        raise TypeError(
            "student_profile must be a StudentProfile "
            "object or dictionary."
        )

    def _bridge_courses(
        self,
        recommended_program,
        predicted_broad_category,
    ):
        """
        Retrieve bridge courses from the specific-program
        mapping, with a broad-field fallback.
        """

        courses = self.bridge_course_map.get(
            recommended_program
        )

        if courses is None:
            courses = self.broad_bridge_course_map.get(
                predicted_broad_category,
                [
                    "Academic Writing",
                    "Digital Literacy",
                    "Study Skills",
                ],
            )

        return format_course_list(courses)

    def recommend(self, student_profile):
        """
        Generate one complete ML-first academic guidance
        result from a learner profile.
        """

        profile = self._profile_dict(student_profile)

        model_ranking = get_model_ranking(profile)

        recommended_program = refine_specific_program(
            profile,
            model_ranking,
        )

        recommended_bridge_course = (
            self._bridge_courses(
                recommended_program,
                model_ranking[
                    "predicted_broad_category"
                ],
            )
        )

        alternative_pathway = get_program_alternative(
            recommended_program,
            model_ranking,
        )

        return RecommendationResult(
            program=recommended_program,
            bridge_courses=recommended_bridge_course,
            alternative_pathway=alternative_pathway,
            source=self.source_label,
            predicted_broad_category=model_ranking[
                "predicted_broad_category"
            ],
        )


RECOMMENDER = AcademicPathwayRecommender()


def recommend_student(student_profile):
    """
    Compatibility wrapper used by the current Streamlit UI.

    The recommendation workflow is owned by the
    AcademicPathwayRecommender class.
    """

    return RECOMMENDER.recommend(
        student_profile
    ).as_tuple()


'''

pattern = re.compile(
    r"def recommend_student\(student_profile\):"
    r".*?(?=\ndef build_explanation\()",
    flags=re.DOTALL,
)

text, count = pattern.subn(
    new_recommender_section.rstrip() + "\n\n",
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "Could not replace recommend_student(). "
        "No changes were saved."
    )

old_call = '''recommended_program, recommended_bridge_course, alternative_pathway, source = recommend_student(
                        profile
                    )'''

new_call = '''recommended_program, recommended_bridge_course, alternative_pathway, source = recommend_student(
                        learner_profile
                    )'''

if old_call not in text:
    raise SystemExit(
        "Could not locate the recommendation call. "
        "No changes were saved."
    )

text = text.replace(
    old_call,
    new_call,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "AcademicPathwayRecommender class added successfully."
)
