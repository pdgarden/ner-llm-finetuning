# ----------------------------------------------------------------------------------------------------------------------
# Import
import json
import random
from datetime import datetime

import openai
from loguru import logger
from pydantic import BaseModel

from src.constants import DATA_DIR, SEED
from src.prompts_synthetic_data_generation import (
    prompt_entities_example,
    prompt_general_instructions,
    prompt_language,
    prompts_synthetic_data_generation,
)
from src.settings import SyntheticSentenceGenerationSettings
from src.type import EntitiesAssets, SyntheticSample, SyntheticSamples

# ----------------------------------------------------------------------------------------------------------------------
# Data structure


class SyntheticSentences(BaseModel):
    sentences: list[str]


# ----------------------------------------------------------------------------------------------------------------------
# Initialize


with (DATA_DIR / "assets" / "entities_names.json").open("r") as f:
    entities_assets = EntitiesAssets.model_validate(json.load(f))

now = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_SAMPLES_FILE = DATA_DIR / f"{now}_synthetic_samples.json"

synthetic_data_generation_settings = SyntheticSentenceGenerationSettings()

llm_client = openai.OpenAI(
    base_url=synthetic_data_generation_settings.llm_client_url,
    api_key=synthetic_data_generation_settings.llm_client_api_key,
)

random.seed(SEED)


# ----------------------------------------------------------------------------------------------------------------------
# Functions


def pick_random_entities(
    entities_assets: EntitiesAssets, num_teams: int, num_players: int
) -> tuple[list[str], list[str]]:
    return (
        random.sample(entities_assets.team_names, num_teams),
        random.sample(entities_assets.player_names, num_players),
    )


def main(settings: SyntheticSentenceGenerationSettings) -> None:
    synthetic_samples: list[SyntheticSample] = []

    for llm_model_id in settings.llm_model_ids:
        for category, prompt_template in prompts_synthetic_data_generation.items():
            for language in settings.languages:
                logger.info(f"Generating synthetic sentences for category: {category} (language: {language})")

                # Generate prompt
                teams, players = pick_random_entities(
                    entities_assets,
                    num_teams=settings.num_samples_team_per_call,
                    num_players=settings.num_samples_player_per_call,
                )
                prompt = prompt_template + prompt_entities_example + prompt_general_instructions + prompt_language
                prompt = prompt.format(
                    team_names_list=", ".join(teams),
                    player_names_list=", ".join(players),
                    language=language,
                )

                # Call LLM
                completion = llm_client.beta.chat.completions.parse(
                    model=llm_model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates synthetic NBA sentences.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=synthetic_data_generation_settings.temperature,
                    top_p=synthetic_data_generation_settings.top_p,
                    response_format=SyntheticSentences,
                )

                # Extract sentences
                sentences = completion.choices[0].message.parsed
                if not sentences:
                    logger.warning(f"No sentences generated for category: {category}")
                    continue

                for sentence in sentences.sentences:
                    synthetic_samples.append(
                        SyntheticSample(
                            sentence=sentence,
                            llm_model_id=llm_model_id,
                            category=category,
                            language=language,
                        )
                    )

    # Save synthetic samples to a JSON file
    output_samples = SyntheticSamples(samples=synthetic_samples)
    logger.info(f"Generated a total of {len(synthetic_samples)} synthetic samples.")
    with OUTPUT_SAMPLES_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_samples.model_dump(), f, indent=4, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    main(synthetic_data_generation_settings)
