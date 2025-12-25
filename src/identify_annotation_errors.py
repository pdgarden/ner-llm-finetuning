# ----------------------------------------------------------------------------------------------------------------------
# Import

import json

import openai
from loguru import logger

from src.constants import SYNTHETIC_DATA_DIR
from src.prompt_identify_error import PROMPT_TEMPLATE
from src.settings import AnnotationErrorIdentificationSettings
from src.type import NERTaskConfidence, SampleErrorRisk, SyntheticSamplesAnnotated

# ----------------------------------------------------------------------------------------------------------------------
# Initialize


annotated_dataset_filepath = SYNTHETIC_DATA_DIR / "annotated_synthetic_samples.json"
output_dataset_filepath = SYNTHETIC_DATA_DIR / "annotated_synthetic_samples_risk.json"

with annotated_dataset_filepath.open("r") as fp:
    annotated_dataset = SyntheticSamplesAnnotated.model_validate_json(fp.read())

settings = AnnotationErrorIdentificationSettings()
llm_client = openai.OpenAI(
    base_url=settings.llm_client_url,
    api_key=settings.llm_client_url,
)


# ----------------------------------------------------------------------------------------------------------------------
# Functions


def main(settings: AnnotationErrorIdentificationSettings) -> None:
    annotated_samples_risk: list[SampleErrorRisk] = []
    # annotated_dataset.samples = annotated_dataset.samples[:100]
    for i, sample in enumerate(annotated_dataset.samples):
        if i % 50 == 0:
            logger.info(f"Sample n°: {i}")

        prompt = PROMPT_TEMPLATE.format(
            sentence=sample.sentence,
            predicted_team_names=sample.annotation.team_names,
            predicted_player_names=sample.annotation.player_names,
        )

        completion = llm_client.beta.chat.completions.parse(
            model=settings.llm_model_id,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=NERTaskConfidence,
        )

        annotation = completion.choices[0].message.parsed
        if not annotation:
            logger.warning(f"No annotation generated for sentence: {sample.sentence}")
            continue

        annotated_sample_risk = SampleErrorRisk(
            sentence=sample.sentence,
            team_names=sample.annotation.team_names,
            player_names=sample.annotation.player_names,
            risk=annotation.risk,
        )

        annotated_samples_risk.append(annotated_sample_risk)

    with output_dataset_filepath.open("w", encoding="utf-8") as f:
        json.dump(
            [sample.model_dump() for sample in annotated_samples_risk],
            f,
            indent=4,
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    main(settings=settings)
