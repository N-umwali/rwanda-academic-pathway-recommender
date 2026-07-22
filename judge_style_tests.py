"""Judge-style integrity checks for the academic pathway recommender.

Run from the project root:
    python judge_style_tests.py

The checks deliberately combine compatible and incompatible learner choices.
They verify that the main recommendation remains eligible for the learner's
General Education route or TVET trade, bridge courses match the final program,
and incompatible preferences cannot override the academic foundation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import app


@dataclass(frozen=True)
class Case:
    name: str
    profile: dict[str, str]
    incompatible_preference: bool = False
    expected_program: str | None = None


def ge_profile(
    pathway: str,
    stream: str,
    best_subject: str,
    career_cluster: str,
    interest_area: str,
) -> dict[str, str]:
    return {
        "EducationType": "General Education",
        "Pathway": pathway,
        "Stream_or_Trade": stream,
        "BestSubject": best_subject,
        "WeakestSubject": "English",
        "InterestArea": interest_area,
        "AverageScoreRange": "70–79%",
        "DigitalSkillLevel": "Intermediate",
        "CareerCluster": career_cluster,
    }


def tvet_profile(
    trade: str,
    career_cluster: str,
    interest_area: str,
) -> dict[str, str]:
    return {
        "EducationType": "TVET",
        "Pathway": "TVET Route",
        "Stream_or_Trade": trade,
        "BestSubject": "Practical Workshop / Execution",
        "WeakestSubject": "Academic Theory",
        "InterestArea": interest_area,
        "AverageScoreRange": "70–79%",
        "DigitalSkillLevel": "Intermediate",
        "CareerCluster": career_cluster,
    }


CASES = [
    Case(
        "Stream 1 supports Medicine",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 1",
            "Biology",
            "Medicine and Surgery",
            "Medicine and Health Sciences",
        ),
    ),
    Case(
        "Stream 1 supports Data Science",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 1",
            "Mathematics",
            "Data Science and Analytics",
            "Data Science, AI, and Machine Learning",
        ),
    ),
    Case(
        "Stream 2 blocks Nursing",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 2",
            "Mathematics",
            "Nursing and Midwifery",
            "Medicine and Health Sciences",
        ),
        incompatible_preference=True,
    ),
    Case(
        "Stream 2 blocks Pharmacy",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 2",
            "Chemistry",
            "Pharmacy and Pharmaceutical Sciences",
            "Medicine and Health Sciences",
        ),
        incompatible_preference=True,
    ),
    Case(
        "Stream 2 supports Civil Engineering",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 2",
            "Physics",
            "Civil Engineering and Construction Technology",
            "Construction and Technical Services",
        ),
    ),
    Case(
        "Stream 2 supports Statistics",
        ge_profile(
            "Mathematics and Sciences",
            "Stream 2",
            "Mathematics",
            "Statistics and Applied Mathematics",
            "Data Science, AI, and Machine Learning",
        ),
    ),
    Case(
        "Arts blocks Medicine",
        ge_profile(
            "Arts and Humanities",
            "Arts and Humanities",
            "History",
            "Medicine and Surgery",
            "Medicine and Health Sciences",
        ),
        incompatible_preference=True,
    ),
    Case(
        "Arts supports Law",
        ge_profile(
            "Arts and Humanities",
            "Arts and Humanities",
            "History",
            "Law and Legal Studies",
            "Law, Governance, and Public Administration",
        ),
    ),
    Case(
        "Languages blocks Engineering",
        ge_profile(
            "Languages",
            "Languages",
            "English",
            "Civil Engineering and Construction Technology",
            "Science, Engineering, and Mathematics",
        ),
        incompatible_preference=True,
    ),
    Case(
        "Languages supports Communication",
        ge_profile(
            "Languages",
            "Languages",
            "English",
            "Communication and Public Relations",
            "Communication, Marketing, and Public Relations",
        ),
    ),
    Case(
        "Building Construction blocks Software override",
        tvet_profile(
            "Building Construction",
            "Software Engineering and Application Development",
            "UI/UX Design and Digital Product Design",
        ),
        incompatible_preference=True,
        expected_program="Civil Engineering and Construction Technology",
    ),
    Case(
        "Electrical Installation blocks Medicine override",
        tvet_profile(
            "Electrical Technology / Electrical Installation",
            "Medicine and Surgery",
            "Medicine and Health Sciences",
        ),
        incompatible_preference=True,
        expected_program="Electrical Technology and Power Systems",
    ),
    Case(
        "Land Surveying continues to GIS",
        tvet_profile(
            "Land Surveying",
            "Geography, GIS and Environmental Planning",
            "Construction and Technical Services",
        ),
        expected_program="Geography, GIS and Environmental Planning",
    ),
    Case(
        "Culinary Arts blocks Journalism override",
        tvet_profile(
            "Culinary Arts",
            "Journalism and Media Studies",
            "Communication, Marketing, and Public Relations",
        ),
        incompatible_preference=True,
        expected_program="Hospitality Management and Culinary Arts",
    ),
    Case(
        "Accounting blocks Data Science override",
        tvet_profile(
            "Accounting",
            "Data Science and Analytics",
            "Data Science, AI, and Machine Learning",
        ),
        incompatible_preference=True,
        expected_program="Accounting and Finance",
    ),
]


def run_case(case: Case) -> list[str]:
    errors: list[str] = []
    result = app.RECOMMENDER.recommend(case.profile)
    program = result.program
    broad = app.SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(program)

    if not broad:
        errors.append(f"unknown recommended program: {program}")
    elif not app.ELIGIBILITY_POLICY.is_broad_category_eligible(
        case.profile,
        broad,
    ):
        errors.append(
            f"ineligible main recommendation: {program} ({broad})"
        )

    expected_bridge = app.format_course_list(
        app.PROGRAM_CATEGORY_TO_BRIDGE_COURSE.get(program)
        or app.MODEL_BROAD_CATEGORY_TO_BRIDGE_COURSE.get(broad)
    )
    if result.bridge_courses != expected_bridge:
        errors.append(
            "bridge courses do not match the final recommended program"
        )

    selected = case.profile["CareerCluster"]
    if case.incompatible_preference and program == selected:
        errors.append(
            "incompatible learner preference overrode academic eligibility"
        )

    if case.expected_program and program != case.expected_program:
        errors.append(
            f"expected {case.expected_program!r}, got {program!r}"
        )

    ranking = app.get_model_ranking(case.profile)
    if ranking["predicted_broad_category"] != broad:
        errors.append(
            "final program broad field does not match the eligible SVM field"
        )

    explanation = app.EXPLAINER.build(
        case.profile,
        result.program,
        result.bridge_courses,
        result.alternative_pathway,
        result.source,
    )
    if case.incompatible_preference:
        transition_words = (
            "future interest",
            "future transition goal",
            "outside the learner's current academic pathway",
        )
        if not any(word in explanation for word in transition_words):
            errors.append(
                "explanation does not identify the incompatible preference as a transition goal"
            )

    return errors


def main() -> None:
    failed = 0
    for case in CASES:
        try:
            errors = run_case(case)
        except Exception as exc:  # noqa: BLE001 - test runner should report all failures
            errors = [f"unexpected exception: {type(exc).__name__}: {exc}"]

        if errors:
            failed += 1
            print(f"FAIL: {case.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS: {case.name}")

    print(f"\n{len(CASES) - failed}/{len(CASES)} checks passed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
