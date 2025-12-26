import openai
import streamlit as st
from annotated_text import annotated_text

from src.prompts_synthetic_data_annotation import ASSISTANT_PROMPT_NER_RETRIEVAL
from src.settings import NERAppSettings
from src.type import SyntheticSampleAnnotation

settings = NERAppSettings()

llm_client = openai.OpenAI(
    base_url=settings.llm_client_url,
    api_key=settings.llm_client_api_key,
)


def extract_entities(text: str) -> SyntheticSampleAnnotation:
    assistant_prompt = ASSISTANT_PROMPT_NER_RETRIEVAL.format(
        expected_json_schema=SyntheticSampleAnnotation.model_json_schema()
    )
    completion = llm_client.beta.chat.completions.parse(
        model=settings.llm_model_id,
        messages=[
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": text},
        ],
        response_format=SyntheticSampleAnnotation,
    )

    # Extract sentences
    parsed = completion.choices[0].message.parsed
    if not parsed:
        error_msg = "No structured output"
        raise ValueError(error_msg)

    return parsed


def format_extracted_entities(  # noqa: C901
    sentence: str, team_names: list[str], player_names: list[str]
) -> list[str | tuple[str, str]]:
    """Process input sentence with extracted entities to generate the format used by streamlit's annotated_text"""
    occurrences = []

    # Process team_names
    for element in team_names:
        start = 0
        while True:
            start = sentence.lower().find(element.lower(), start)
            if start == -1:
                break
            end = start + len(element)
            occurrences.append((start, end, element, "team"))
            start = end

    # Process player_names
    for element in player_names:
        start = 0
        while True:
            start = sentence.lower().find(element.lower(), start)
            if start == -1:
                break
            end = start + len(element)
            occurrences.append((start, end, element, "player"))
            start = end

    # Sort occurrences by start index
    occurrences.sort(key=lambda x: x[0])

    # Build the result
    result = []
    current_pos = 0
    for start, end, element, identifier in occurrences:
        # Add the part before this occurrence if it's not just whitespace
        if current_pos < start:
            substring = sentence[current_pos:start]
            if substring.strip():  # if substring is not just whitespace
                result.append(substring)
        result.append((element, identifier))
        current_pos = end

    # Add the remaining part of the sentence if it's not just whitespace
    if current_pos < len(sentence):
        remaining = sentence[current_pos:]
        if remaining.strip():
            result.append(remaining)

    return result


st.title("Named Entity Recognition - NBA teams and players")
input_text = st.text_input("Sentence to process")
perform_ner = st.button("Execute")


if perform_ner:
    extracted_entities = extract_entities(text=input_text)
    sentence_annotated = format_extracted_entities(
        sentence=input_text,
        team_names=extracted_entities.team_names,
        player_names=extracted_entities.player_names,
    )
    annotated_text(sentence_annotated)
    st.divider()
    st.code(extracted_entities.model_dump_json(indent=3))
