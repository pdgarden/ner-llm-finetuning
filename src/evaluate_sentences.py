# ----------------------------------------------------------------------------------------------------------------------
# Import
import json
from datetime import datetime
from time import perf_counter

import openai
from loguru import logger

from src.constants import ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE, EVALUATED_DATA_DIR
from src.prompts_synthetic_data_annotation import ASSISTANT_PROMPT_NER_RETRIEVAL
from src.settings import SentenceEvaluationSettings
from src.type import SampleEvaluated, SamplesEvaluated, SplitSyntheticSamplesAnnotated, SyntheticSampleAnnotation

# ----------------------------------------------------------------------------------------------------------------------
# Initialize


with ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE.open("r", encoding="utf-8") as f:
    test_samples = SplitSyntheticSamplesAnnotated.model_validate(json.load(f)).test
    # test_samples.samples = test_samples.samples[:30]  # Temporary limit for faster testing

now = datetime.now().strftime("%Y%m%d_%H%M%S")
now = ""
OUTPUT_EVALUATION_FILE = EVALUATED_DATA_DIR / f"{now}_evaluated_samples.json"


annotated_data_generation_settings = SentenceEvaluationSettings()

llm_client = openai.OpenAI(
    base_url=annotated_data_generation_settings.llm_client_url,
    api_key=annotated_data_generation_settings.llm_client_api_key,
)

# ----------------------------------------------------------------------------------------------------------------------
# Functions


def main(settings: SentenceEvaluationSettings) -> None:
    evaluated_samples: list[SampleEvaluated] = []

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            logger.info(f"Evaluating sample n°{i}")

        assistant_prompt = ASSISTANT_PROMPT_NER_RETRIEVAL.format(
            expected_json_schema=SyntheticSampleAnnotation.model_json_schema()
        )

        start_time = perf_counter()
        try:
            completion = llm_client.beta.chat.completions.parse(
                model=settings.llm_model_id,
                messages=[
                    {"role": "assistant", "content": assistant_prompt},
                    {"role": "user", "content": sample.sentence},
                ],
                temperature=settings.temperature,
                response_format=SyntheticSampleAnnotation,
                max_tokens=settings.max_tokens,
            )

            # Extract sentences
            parsed = completion.choices[0].message.parsed

            if not parsed:
                raise ValueError  # noqa: TRY301

        except Exception as exc:
            logger.warning(f"Exception while generating prediction for sample n° {i}: {exc}")
            parsed = SyntheticSampleAnnotation(team_names=["FAILED_GENERATION"], player_names=["FAILED_GENERATION"])

        processing_duration_seconds = perf_counter() - start_time
        predicted_team_names = parsed.team_names
        predicted_player_names = parsed.player_names

        evaluated_sample = SampleEvaluated(
            sample_annotated=sample,
            processing_duration_seconds=processing_duration_seconds,
            predicted_team_names=predicted_team_names,
            predicted_player_names=predicted_player_names,
        )
        evaluated_samples.append(evaluated_sample)

    samples_evaluated = SamplesEvaluated(
        samples_evaluated=evaluated_samples,
        evaluation_settings=settings,
        prompt_template=ASSISTANT_PROMPT_NER_RETRIEVAL,
    )

    with OUTPUT_EVALUATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(samples_evaluated.model_dump(), f, indent=4, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------------------------------
# Entry point
if __name__ == "__main__":
    main(annotated_data_generation_settings)
