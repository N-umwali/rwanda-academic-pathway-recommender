from pathlib import Path
import re

path = Path("app.py")
text = path.read_text(encoding="utf-8")

if "class RecommendationExplainer:" in text:
    print("RecommendationExplainer is already present.")
    raise SystemExit


new_explainer_section = '''class RecommendationExplainer:
    """
    Convert model and SHAP results into clear guidance
    for learners and academic advisors.

    This class separates explanation responsibilities
    from the Streamlit interface and recommendation service.
    """

    @staticmethod
    def _profile_dict(profile):
        """
        Accept either a StudentProfile object or the existing
        model-compatible dictionary.
        """

        if isinstance(profile, StudentProfile):
            return profile.to_dict()

        if isinstance(profile, dict):
            return dict(profile)

        raise TypeError(
            "profile must be a StudentProfile object "
            "or dictionary."
        )

    @staticmethod
    def _clean_value(value, fallback="Not specified"):
        """Return a readable profile value."""

        text = str(value or "").strip()
        return text if text else fallback

    @staticmethod
    def _format_influence_list(explanation_rows):
        """
        Convert the three strongest SHAP-supported profile
        factors into learner-friendly wording.
        """

        most_influential = explanation_rows[:3]

        if not most_influential:
            return (
                "Your education background, strengths, interests, "
                "and career direction were considered together "
                "when preparing this guidance."
            )

        influence_items = [
            f"**{row['display_name']}**"
            for row in most_influential
        ]

        if len(influence_items) == 1:
            influence_list = influence_items[0]

        elif len(influence_items) == 2:
            influence_list = (
                f"{influence_items[0]} and "
                f"{influence_items[1]}"
            )

        else:
            influence_list = (
                f"{influence_items[0]}, "
                f"{influence_items[1]}, and "
                f"{influence_items[2]}"
            )

        return (
            "The parts of your profile that had the greatest "
            f"influence on this guidance were your "
            f"{influence_list}."
        )

    def build(
        self,
        profile,
        program,
        bridge,
        alternative,
        source,
    ):
        """
        Create the complete learner-facing explanation.

        The visible wording remains nontechnical, while the
        influential factors come from genuine SHAP values.
        """

        profile = self._profile_dict(profile)

        education_type = self._clean_value(
            profile.get("EducationType")
        )
        pathway = self._clean_value(
            profile.get("Pathway")
        )
        stream_or_trade = self._clean_value(
            profile.get("Stream_or_Trade")
        )
        strongest_area = self._clean_value(
            profile.get("BestSubject")
        )
        support_area = self._clean_value(
            profile.get("WeakestSubject")
        )
        interest_area = self._clean_value(
            profile.get("InterestArea")
        )
        career_direction = self._clean_value(
            profile.get("CareerCluster")
        )
        score_range = self._clean_value(
            profile.get("AverageScoreRange")
        )

        model_ranking = get_model_ranking(profile)

        explanation_rows = get_grouped_model_explanation(
            profile,
            model_ranking,
        )

        influence_reason = self._format_influence_list(
            explanation_rows
        )

        if education_type == "TVET":
            recommendation_reason = (
                f"**{program}** is recommended because your "
                f"TVET training in **{stream_or_trade}** "
                f"provides a relevant foundation for this "
                f"academic direction. Your strongest competency, "
                f"**{strongest_area}**, also supports further "
                f"study in this area."
            )

            expected_interest = self._clean_value(
                TVET_TRADE_TO_INTEREST_AREA.get(
                    stream_or_trade,
                    "",
                ),
                "",
            )

            expected_career = self._clean_value(
                TVET_TRADE_TO_CAREER_CLUSTER.get(
                    stream_or_trade,
                    "",
                ),
                "",
            )

            interest_is_different = (
                expected_interest
                and interest_area.casefold()
                != expected_interest.casefold()
            )

            career_is_different = (
                expected_career
                and career_direction.casefold()
                != expected_career.casefold()
            )

            if interest_is_different or career_is_different:
                preference_reason = (
                    f"You also selected **{interest_area}** as "
                    f"an interest area and "
                    f"**{career_direction}** as a career "
                    f"direction. These choices may differ from "
                    f"your current TVET trade. This recommendation "
                    f"therefore gives greater attention to the "
                    f"training and practical skills you already "
                    f"have. An academic advisor can help you plan "
                    f"a transition when the new direction is your "
                    f"main goal."
                )

            else:
                preference_reason = (
                    f"Your interest in **{interest_area}** and "
                    f"your preferred career direction, "
                    f"**{career_direction}**, are also consistent "
                    f"with this recommendation."
                )

        else:
            recommendation_reason = (
                f"**{program}** is recommended because your "
                f"background in **{pathway}**, particularly "
                f"**{stream_or_trade}**, provides a relevant "
                f"starting point for this direction. Your "
                f"strongest subject, **{strongest_area}**, also "
                f"supports the knowledge and skills commonly "
                f"required in this field."
            )

            preference_reason = (
                f"Your interest in **{interest_area}** and your "
                f"preferred career direction, "
                f"**{career_direction}**, were also considered "
                f"when identifying this recommendation."
            )

        if (
            strongest_area.casefold()
            == support_area.casefold()
            and strongest_area != "Not specified"
        ):
            support_reason = (
                f"You selected **{strongest_area}** as both your "
                f"strongest area and the area where you need "
                f"support. Review this choice to make sure your "
                f"profile accurately reflects your learning needs."
            )

        else:
            support_reason = (
                f"You identified **{support_area}** as an area "
                f"needing support. Strengthening this area may "
                f"improve your overall preparation for further "
                f"study."
            )

        preparation_reason = (
            f"To prepare for **{program}**, the recommended "
            f"bridge courses are **{bridge}**. These courses can "
            f"help strengthen the knowledge and practical skills "
            f"commonly needed in this program."
        )

        if score_range == "50–59%":
            readiness_reason = (
                "Your average score range is **50–59%**. This "
                "recommendation should be treated as a possible "
                "pathway rather than a confirmed admission match. "
                "Completing the suggested bridge preparation and "
                "speaking with an academic advisor will be "
                "especially important."
            )

        elif score_range == "60–69%":
            readiness_reason = (
                "Your average score range is **60–69%**. "
                "Additional preparation in the recommended bridge "
                "areas may strengthen your readiness and improve "
                "your options."
            )

        else:
            readiness_reason = (
                f"Your average score range of **{score_range}** "
                f"provides a useful starting point for this "
                f"direction. Official admission requirements "
                f"should still be confirmed with the institution."
            )

        return (
            f"{recommendation_reason}\n\n"
            f"{preference_reason}\n\n"
            f"{influence_reason}\n\n"
            f"{support_reason}\n\n"
            f"{preparation_reason}\n\n"
            f"{readiness_reason}\n\n"
            f"**Another pathway to consider:** "
            f"{alternative}\n\n"
            f"This recommendation is intended to support "
            f"exploration and discussion with an academic "
            f"advisor. It is not an official admission decision."
        )


EXPLAINER = RecommendationExplainer()


def build_explanation(
    profile,
    program,
    bridge,
    alternative,
    source,
):
    """
    Compatibility wrapper used by the current Streamlit UI.

    Explanation generation is owned by the
    RecommendationExplainer class.
    """

    return EXPLAINER.build(
        profile=profile,
        program=program,
        bridge=bridge,
        alternative=alternative,
        source=source,
    )


'''

pattern = re.compile(
    r"def build_explanation\(\n"
    r"    profile,\n"
    r"    program,\n"
    r"    bridge,\n"
    r"    alternative,\n"
    r"    source,\n"
    r"\):"
    r".*?(?=\ndef make_guidance_report\()",
    flags=re.DOTALL,
)

text, count = pattern.subn(
    new_explainer_section.rstrip() + "\n\n",
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "Could not replace build_explanation(). "
        "No changes were saved."
    )

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "RecommendationExplainer class added successfully."
)
