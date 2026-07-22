from pathlib import Path
import re
import textwrap

path = Path("app.py")
text = path.read_text(encoding="utf-8")

if "class RecommendationExplainer:" in text:
    raise SystemExit(
        "RecommendationExplainer already exists. "
        "No changes were made."
    )

pattern = re.compile(
    r"def build_explanation\(\n"
    r"    profile,\n"
    r"    program,\n"
    r"    bridge,\n"
    r"    alternative,\n"
    r"    source,\n"
    r"\):"
    r".*?(?=\n\ndef make_guidance_report\()",
    flags=re.DOTALL,
)

match = pattern.search(text)

if not match:
    raise SystemExit(
        "The existing build_explanation function "
        "could not be found."
    )

original_function = match.group(0)

method_function = original_function.replace(
    "def build_explanation(\n"
    "    profile,\n",
    "def build(\n"
    "    self,\n"
    "    profile,\n",
    1,
)

indented_method = textwrap.indent(
    method_function,
    "    ",
)

replacement = (
    'class RecommendationExplainer:\n'
    '    """\n'
    '    Generate learner-friendly explanations using\n'
    '    genuine SHAP-supported model influences.\n'
    '\n'
    '    The class separates explanation responsibilities\n'
    '    from the Streamlit user interface.\n'
    '    """\n\n'
    f'{indented_method}\n\n\n'
    'EXPLAINER = RecommendationExplainer()\n\n\n'
    'def build_explanation(\n'
    '    profile,\n'
    '    program,\n'
    '    bridge,\n'
    '    alternative,\n'
    '    source,\n'
    '):\n'
    '    """Compatibility wrapper for the existing UI."""\n\n'
    '    return EXPLAINER.build(\n'
    '        profile=profile,\n'
    '        program=program,\n'
    '        bridge=bridge,\n'
    '        alternative=alternative,\n'
    '        source=source,\n'
    '    )'
)

text = (
    text[:match.start()]
    + replacement
    + text[match.end():]
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "RecommendationExplainer class added successfully."
)
