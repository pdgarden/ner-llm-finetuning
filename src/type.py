import hashlib
import statistics
from typing import Literal

from pydantic import BaseModel, computed_field, field_validator

from src.settings import SentenceEvaluationSettings


class EntitiesAssets(BaseModel):
    team_names: list[str]
    player_names: list[str]


class SyntheticSampleAnnotation(BaseModel):
    team_names: list[str]
    player_names: list[str]


class SyntheticSample(BaseModel):
    sentence: str
    llm_model_id: str
    category: str
    language: str


class SyntheticSampleAnnotated(BaseModel):
    sentence: str
    llm_model_id: str
    category: str
    language: str
    annotation: SyntheticSampleAnnotation


class SyntheticSamples(BaseModel):
    samples: list[SyntheticSample]


class SyntheticSamplesAnnotated(BaseModel):
    samples: list[SyntheticSampleAnnotated]


class SampleEvaluated(BaseModel):
    sample_annotated: SyntheticSampleAnnotated
    processing_duration_seconds: float
    predicted_team_names: list[str]
    predicted_player_names: list[str]

    @field_validator("processing_duration_seconds", mode="after")
    @classmethod
    def round(cls, value: float) -> float:
        return round(value, 4)

    @computed_field
    @property
    def is_correct_team_names(self) -> bool:
        return set(self.sample_annotated.annotation.team_names) == set(self.predicted_team_names)

    @computed_field
    @property
    def is_correct_player_names(self) -> bool:
        return set(self.sample_annotated.annotation.player_names) == set(self.predicted_player_names)

    @computed_field
    @property
    def is_fully_correct(self) -> bool:
        return self.is_correct_team_names and self.is_correct_player_names


class SamplesEvaluated(BaseModel):
    samples_evaluated: list[SampleEvaluated]
    evaluation_settings: SentenceEvaluationSettings
    prompt_template: str

    @computed_field
    @property
    def accuracy_team_names(self) -> float:
        if not self.samples_evaluated:
            return 0.0
        return round(statistics.mean([s.is_correct_team_names for s in self.samples_evaluated]), 4)

    @computed_field
    @property
    def accuracy_player_names(self) -> float:
        if not self.samples_evaluated:
            return 0.0
        return round(statistics.mean([s.is_correct_player_names for s in self.samples_evaluated]), 4)

    @computed_field
    @property
    def accuracy_fully_correct(self) -> float:
        if not self.samples_evaluated:
            return 0.0
        return round(statistics.mean([s.is_fully_correct for s in self.samples_evaluated]), 4)

    @computed_field
    @property
    def mean_processing_duration_seconds(self) -> float:
        if not self.samples_evaluated:
            return 0.0
        return round(statistics.mean(s.processing_duration_seconds for s in self.samples_evaluated), 2)

    @computed_field
    @property
    def prompt_md5(self) -> str:
        return hashlib.md5(self.prompt_template.encode("utf-8")).hexdigest()  # noqa: S324


class NERTaskConfidence(BaseModel):
    risk: Literal["low", "medium", "high"]


class SampleErrorRisk(BaseModel):
    sentence: str
    team_names: list[str]
    player_names: list[str]
    risk: Literal["low", "medium", "high"]


class SplitSyntheticSamplesAnnotated(BaseModel):
    train: list[SyntheticSampleAnnotated]
    test: list[SyntheticSampleAnnotated]
