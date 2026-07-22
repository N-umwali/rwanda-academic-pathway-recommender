from pathlib import Path
import re

path = Path("app.py")
text = path.read_text(encoding="utf-8")

# ---------------------------------------------------------
# 1. Use the learner's TVET trade—not an unrelated selected
#    career—as the CareerCluster input sent to the model.
# ---------------------------------------------------------

old_model_block = '''        profile["InterestArea"] = "Technical Trades"

        selected_specific_program = str(
            profile.get("CareerCluster", "")
        ).strip()
        trade_aligned_program = TVET_TRADE_TO_CAREER_CLUSTER.get(
            original_stream_or_trade,
            selected_specific_program,
        )
        program_for_model = (
            selected_specific_program
            if selected_specific_program in MODEL_CAREER_CLUSTER_MAP
            else trade_aligned_program
        )
        profile["CareerCluster"] = MODEL_CAREER_CLUSTER_MAP.get(
            program_for_model,
            "Engineering and Infrastructure",
        )
'''

new_model_block = '''        profile["InterestArea"] = "Technical Trades"

        trade_aligned_program = (
            TVET_TRADE_TO_CAREER_CLUSTER.get(
                original_stream_or_trade,
                "Mechanical and Manufacturing Engineering",
            )
        )

        profile["CareerCluster"] = (
            MODEL_CAREER_CLUSTER_MAP.get(
                trade_aligned_program,
                "Engineering and Infrastructure",
            )
        )
'''

if old_model_block not in text:
    raise SystemExit(
        "The TVET model-input block was not found. "
        "No changes were saved."
    )

text = text.replace(
    old_model_block,
    new_model_block,
    1,
)

# ---------------------------------------------------------
# 2. Add a helper that compares the selected career with
#    the learner's actual TVET trade.
# ---------------------------------------------------------

helper = '''def get_tvet_alignment_context(student_profile):
    """
    Compare the learner's TVET trade with the selected
    career direction.
    """

    trade = str(
        student_profile.get("Stream_or_Trade", "")
    ).strip()

    selected_program = str(
        student_profile.get("CareerCluster", "")
    ).strip()

    trade_program = TVET_TRADE_TO_CAREER_CLUSTER.get(
        trade
    )

    trade_broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
            trade_program
        )
    )

    selected_broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
            selected_program
        )
    )

    return {
        "trade_program": trade_program,
        "trade_broad_category": trade_broad_category,
        "selected_program": selected_program,
        "selected_broad_category": selected_broad_category,
        "same_field": bool(
            trade_broad_category
            and selected_broad_category
            and trade_broad_category
            == selected_broad_category
        ),
    }


'''

if "def get_tvet_alignment_context" not in text:
    marker = "def refine_specific_program(student_profile, model_ranking):"

    if marker not in text:
        raise SystemExit(
            "refine_specific_program() was not found."
        )

    text = text.replace(
        marker,
        helper + marker,
        1,
    )

# ---------------------------------------------------------
# 3. Replace refinement so an unrelated selected career
#    cannot override the TVET trade.
# ---------------------------------------------------------

new_refine_function = '''def refine_specific_program(student_profile, model_ranking):
    """
    Refine the SVM broad field into a specific program.

    General Education profiles may use the learner-selected
    program when it is supported by the model.

    TVET profiles remain grounded in the current trade.
    A selected career may refine the main recommendation only
    when it belongs to the same broad field as that trade.
    """

    predicted_broad_category = model_ranking[
        "predicted_broad_category"
    ]

    ranked_broad_categories = model_ranking[
        "ranked_broad_categories"
    ]

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

        # A learner-selected program may refine the result
        # only when it belongs to the same broad field as
        # the learner's existing TVET trade.
        if (
            alignment["same_field"]
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

        # Otherwise, use the direct TVET trade continuation
        # when the model supports that broad field.
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

        # Final fallback remains the SVM-predicted field.
        return BROAD_CATEGORY_DEFAULT_PROGRAM.get(
            predicted_broad_category,
            trade_program or predicted_broad_category,
        )

    preferred_program = get_preferred_specific_program(
        student_profile
    )

    preferred_broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(
            preferred_program
        )
    )

    if preferred_program in SPECIFIC_PROGRAM_TO_BROAD_CATEGORY:
        if (
            preferred_broad_category
            in ranked_broad_categories[:3]
            or broad_categories_are_related(
                preferred_broad_category,
                predicted_broad_category,
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
            == predicted_broad_category
        ):
            return candidate

    return BROAD_CATEGORY_DEFAULT_PROGRAM.get(
        predicted_broad_category,
        preferred_program or predicted_broad_category,
    )


'''

pattern = re.compile(
    r"def refine_specific_program\(student_profile, model_ranking\):"
    r".*?(?=\ndef get_program_alternative\()",
    flags=re.DOTALL,
)

text, replacements = pattern.subn(
    new_refine_function.rstrip() + "\n\n",
    text,
    count=1,
)

if replacements != 1:
    raise SystemExit(
        "The refinement function could not be replaced. "
        "No changes were saved."
    )

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "TVET trade-alignment logic updated successfully."
)
