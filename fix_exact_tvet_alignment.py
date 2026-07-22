from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")

old_return = '''        "same_field": bool(
            trade_broad_category
            and selected_broad_category
            and trade_broad_category
            == selected_broad_category
        ),
'''

new_return = '''        "same_specialisation": bool(
            trade_program
            and selected_program
            and trade_program == selected_program
        ),
        "same_field": bool(
            trade_broad_category
            and selected_broad_category
            and trade_broad_category
            == selected_broad_category
        ),
'''

if old_return not in text:
    raise SystemExit(
        "Could not locate the TVET alignment return block."
    )

text = text.replace(
    old_return,
    new_return,
    1,
)

old_condition = '''            alignment["same_field"]
            and selected_program
'''

new_condition = '''            alignment["same_specialisation"]
            and selected_program
'''

if old_condition not in text:
    raise SystemExit(
        "Could not locate the TVET refinement condition."
    )

text = text.replace(
    old_condition,
    new_condition,
    1,
)

text = text.replace(
    "when it belongs to the same broad field as that trade.",
    "when it matches the specialisation aligned with that trade.",
    1,
)

text = text.replace(
    "# only when it belongs to the same broad field as",
    "# only when it matches the direct specialisation for",
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "Exact TVET specialisation alignment added successfully."
)
