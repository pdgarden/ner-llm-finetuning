# README

TODO: Clean README

# Steps:

# Settings

Can be configured through environment variables, see [src/settings.py](src/settings.py)

# Commands

1. `uv sync`
2. `uv run python -m src.generate_synthetic_sentences`
3. `uv run python -m src.annotate_sentences`
4. `uv run python -m src.evaluate_sentences`
5. `uv run python -m src.identify_errors` (to help manually cleaning the dataset)
6. `uv run python -m streamlit run src/app_ner.py`




Run app:

# Improvements

- Add variability in synthetic dataset (upper / lower case)
