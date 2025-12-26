# ----------------------------------------------------------------------------------------------------------------------
# Import

import json
import random

from src.constants import ANNOTATED_SYNTHETIC_DATA_FILE, ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE, SEED
from src.settings import SplitAndPreprocessDatasetSettings
from src.type import SplitSyntheticSamplesAnnotated, SyntheticSamplesAnnotated

# ----------------------------------------------------------------------------------------------------------------------
# Initialize


with ANNOTATED_SYNTHETIC_DATA_FILE.open("r", encoding="utf-8") as f:
    annotated_samples = SyntheticSamplesAnnotated.model_validate(json.load(f))

settings = SplitAndPreprocessDatasetSettings()

random.seed(SEED)

# ----------------------------------------------------------------------------------------------------------------------
# Functions


def main(settings: SplitAndPreprocessDatasetSettings) -> None:
    for sample in annotated_samples.samples:
        clean_teams = []
        for team_name in sample.annotation.team_names:
            if random.random() > settings.team_lowercase_rate:
                clean_teams.append(team_name)
            else:
                tn = team_name.lower()
                clean_teams.append(tn)
                sample.sentence = sample.sentence.replace(team_name, tn)

        clean_players = []
        for player_name in sample.annotation.player_names:
            if random.random() > settings.player_lowercase_rate:
                clean_players.append(player_name)
            else:
                pn = player_name.lower()
                clean_players.append(pn)
                sample.sentence = sample.sentence.replace(player_name, pn)

        sample.annotation.team_names = clean_teams
        sample.annotation.player_names = clean_players

    random.shuffle(annotated_samples.samples)

    split_idx = int(len(annotated_samples.samples) * settings.train_test_split_ratio)
    train_samples = annotated_samples.samples[:split_idx]
    test_samples = annotated_samples.samples[split_idx:]

    processed_sample = SplitSyntheticSamplesAnnotated(
        train=train_samples,
        test=test_samples,
    )

    with ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE.open("w", encoding="utf-8") as fp:
        json.dump(processed_sample.model_dump(), fp, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    main(settings=settings)
