# ----------------------------------------------------------------------------------------------------------------------
# Import
import json
import random
from datetime import datetime

import openai
from loguru import logger

from src.constants import SEED, SYNTHETIC_DATA_DIR, SYNTHETIC_DATA_FILE, TRAIN_TEST_SPLIT_RATIO
from src.prompts_synthetic_data_annotation import ASSISTANT_PROMPT_NER_RETRIEVAL
from src.settings import SyntheticSentenceGenerationSettings
from src.type import SyntheticSampleAnnotated, SyntheticSampleAnnotation, SyntheticSamples, SyntheticSamplesAnnotated

# ----------------------------------------------------------------------------------------------------------------------
# Initialize


with SYNTHETIC_DATA_FILE.open("r", encoding="utf-8") as f:
    synthetic_samples = SyntheticSamples.model_validate(json.load(f))
    # synthetic_samples.samples = synthetic_samples.samples[:10]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_SAMPLES_FILE = SYNTHETIC_DATA_DIR / f"{now}_annotated_synthetic_samples.json"
OUTPUT_SAMPLES_TRAIN_FILE = SYNTHETIC_DATA_DIR / f"{now}_annotated_synthetic_samples_train.json"
OUTPUT_SAMPLES_TEST_FILE = SYNTHETIC_DATA_DIR / f"{now}_annotated_synthetic_samples_test.json"


annotated_data_generation_settings = SyntheticSentenceGenerationSettings()

llm_client = openai.OpenAI(
    base_url=annotated_data_generation_settings.llm_client_url,
    api_key=annotated_data_generation_settings.llm_client_api_key,
)

random.seed(SEED)

# ----------------------------------------------------------------------------------------------------------------------
# Functions


def main(settings: SyntheticSentenceGenerationSettings) -> None:
    annotated_samples: list[SyntheticSampleAnnotated] = []

    for i, sample in enumerate(synthetic_samples.samples):
        if i % 20 == 0:
            logger.info(f"Annotating sample n°{i}")

        assistant_prompt = ASSISTANT_PROMPT_NER_RETRIEVAL.format(
            expected_json_schema=SyntheticSampleAnnotation.model_json_schema()
        )

        completion = llm_client.beta.chat.completions.parse(
            model=settings.llm_model_id,
            messages=[
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": sample.sentence},
            ],
            temperature=settings.temperature,
            response_format=SyntheticSampleAnnotation,
        )

        # Extract sentences
        annotation = completion.choices[0].message.parsed
        if not annotation:
            logger.warning(f"No annotation generated for sentence: {sample.sentence}")
            continue

        # Parse the annotation content (assuming it's in JSON format)

        annotated_sample = SyntheticSampleAnnotated(
            sentence=sample.sentence,
            llm_model_id=sample.llm_model_id,
            category=sample.category,
            language=sample.language,
            annotation=annotation,
        )

        annotated_samples.append(annotated_sample)

    annotated_synthetic_samples = SyntheticSamplesAnnotated(samples=annotated_samples)

    # Split into train and test
    categories = list(set(sample.category for sample in annotated_samples))
    random.shuffle(categories)
    train_categories = categories[: int(TRAIN_TEST_SPLIT_RATIO * len(categories))]
    test_categories = categories[int(TRAIN_TEST_SPLIT_RATIO * len(categories)) :]
    train_samples = [sample for sample in annotated_samples if sample.category in train_categories]
    test_samples = [sample for sample in annotated_samples if sample.category in test_categories]

    # Save annotated samples
    with OUTPUT_SAMPLES_FILE.open("w", encoding="utf-8") as f:
        json.dump(annotated_synthetic_samples.model_dump(), f, indent=4, ensure_ascii=False)

    with OUTPUT_SAMPLES_TRAIN_FILE.open("w", encoding="utf-8") as f:
        json.dump(SyntheticSamplesAnnotated(samples=train_samples).model_dump(), f, indent=4, ensure_ascii=False)
    with OUTPUT_SAMPLES_TEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(SyntheticSamplesAnnotated(samples=test_samples).model_dump(), f, indent=4, ensure_ascii=False)

    logger.info(f"Annotated samples saved to {OUTPUT_SAMPLES_FILE}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    main(settings=annotated_data_generation_settings)
