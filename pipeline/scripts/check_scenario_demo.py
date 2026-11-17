"""Exercise the tabular demonstration with its synthetic case."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


app = AppTest.from_file(str(Path(__file__).with_name("scenario_demo.py")), default_timeout=90).run()
assert not app.exception, app.exception
tables = [element.value for element in app.dataframe]
candidate_table = next(table for table in tables if "selected" in table.columns)
assert int(candidate_table["selected"].sum()) == 1
app.button[0].click().run()
assert not app.exception, app.exception
assert sum("SHAP value (neural logit)" in e.value.columns for e in app.dataframe) == 3
print("Synthetic monitoring inputs, candidate ranking and individual SHAP passed.")
