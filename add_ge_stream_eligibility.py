from pathlib import Path
import re

path = Path("app.py")
text = path.read_text(encoding="utf-8")

if "STREAM_2_BLOCKED_BROAD_CATEGORIES" in text:
    print("Stream 2 eligibility filtering is already present.")
    raise SystemExit


eligibility_helpers = '''STREAM_2_BLOCKED_BROAD_CATEGORIES = {
    "Medicine and Surgery",
    "Nursing and Midwifery",
    "Pharmacy and Pharmaceutical Sciences",
    "Biomedical Laboratory Sciences",
    "Biotechnology and Applied Biosciences",
}


def is_broad_category_eligible_for_profile(
    student_profile,
    broad_category,
):
    """
    Check whether a broad academic field is consistent
    with the learner's General Education stream.

    Stream 2 does not include Biology and Chemistry.
    Health and laboratory-science fields are therefore
    not presented as direct main recommendations.
    """

    if (
        student_profile.get("EducationType")
        != "General Education"
    ):
        return True

    stream = str(
        student_profile.get("Stream_or_Trade", "")
    ).strip()

    if (
        stream == "Stream 2"
        and broad_category
        in STREAM_2_BLOCKED_BROAD_CATEGORIES
    ):
        return False

    return True


def is_program_eligible_for_profile(
    student_profile,
    program,
):
    """Check eligibility for one specific program."""

    broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
            program
        )
    )

    if not broad_category:
        return False

    return is_broad_category_eligible_for_profile(
        student_profile,
        broad_category,
    )


'''


marker = "def refine_specific_program(student_profile, model_ranking):"

if marker not in text:
    raise SystemExit(
        "Could not find refine_specific_program()."
    )

text = text.replace(
    marker,
    eligibility_helpers + marker,
    1,
)


new_refine_function = '''def refine_specific_program(student_profile, model_ranking):
    """
    Refine the SVM broad field into a specific program.

    TVET profiles remain grounded in their current trade.

    General Education profiles are filtered using stream
    eligibility before a specific program is selected.
    """

    predicted_broad_category = model_ranking[
        "predicted_broad_category"
    ]

    ranked_broad_categories = model_ranking[
        "ranked_broad_categories"
    ]

    # -----------------------------------------------------
    # TVET recommendation refinement
    # -----------------------------------------------------

    if student_profile.get("EducationType") == "TVET":
        alignment = get_tvet_alignment_context(
            student_profile
        )

        trade_program = alignment[
            "trade_program"
        ]

        trade_broad_category = alignment[
            "trade_broad_category"
        ]

        selected_program = alignment[
            "selected_program"
        ]

        selected_broad_category = alignment[
            "selected_broad_category"
        ]

        if (
            alignment["same_specialisation"]
            and selected_program
            in SPECIFIC_PROGRAM_TO_BROAD_CATEGORY
            and (
                selected_broad_category
                in ranked_broad_categories[:3]
                or broad_categories_are_related(
                    selected_broad_category,
                    predicted_broad_category,
                )
            )
        ):
            return selected_program

        if (
            trade_program
            and trade_broad_category
            and (
                trade_broad_category
                in ranked_broad_categories[:3]
                or broad_categories_are_related(
                    trade_broad_category,
                    predicted_broad_category,
                )
            )
        ):
            return trade_program

        return BROAD_CATEGORY_DEFAULT_PROGRAM.get(
            predicted_broad_category,
            trade_program or predicted_broad_category,
        )

    # -----------------------------------------------------
    # General Education eligibility filtering
    # -----------------------------------------------------

    eligible_ranked_categories = [
        broad_category
        for broad_category in ranked_broad_categories
        if is_broad_category_eligible_for_profile(
            student_profile,
            broad_category,
        )
    ]

    if eligible_ranked_categories:
        eligible_predicted_category = (
            eligible_ranked_categories[0]
        )
    else:
        eligible_predicted_category = (
            predicted_broad_category
        )

    preferred_program = get_preferred_specific_program(
        student_profile
    )

    preferred_broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
            preferred_program
        )
    )

    if (
        preferred_program
        in SPECIFIC_PROGRAM_TO_BROAD_CATEGORY
        and is_program_eligible_for_profile(
            student_profile,
            preferred_program,
        )
        and (
            preferred_broad_category
            in eligible_ranked_categories[:3]
            or broad_categories_are_related(
                preferred_broad_category,
                eligible_predicted_category,
            )
        )
    ):
        return preferred_program

    interest_area = student_profile.get(
        "InterestArea"
    )

    interest_candidates = (
        INTEREST_AREA_TO_CAREER_CLUSTERS.get(
            interest_area,
            [],
        )
    )

    for candidate in interest_candidates:
        candidate_broad_category = (
            SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
                candidate
            )
        )

        if (
            candidate_broad_category
            == eligible_predicted_category
            and is_program_eligible_for_profile(
                student_profile,
                candidate,
            )
        ):
            return candidate

    return BROAD_CATEGORY_DEFAULT_PROGRAM.get(
        eligible_predicted_category,
        preferred_program
        or eligible_predicted_category,
    )


'''

pattern = re.compile(
    r"def refine_specific_program\(student_profile, model_ranking\):"
    r".*?(?=\ndef get_program_alternative\()",
    flags=re.DOTALL,
)

text, replacement_count = pattern.subn(
    new_refine_function.rstrip() + "\n\n",
    text,
    count=1,
)

if replacement_count != 1:
    raise SystemExit(
        "Could not replace refine_specific_program(). "
        "No changes were saved."
    )

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "General Education stream eligibility added successfully."
)
