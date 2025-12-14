from pydantic_settings import BaseSettings


class SyntheticSentenceGenerationSettings(BaseSettings):
    llm_model_id: str = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL"
    llm_client_url: str = "http://localhost:11434/v1"  # default to ollama
    llm_client_api_key: str = "any"
    temperature: float = 1.0
    top_p: float = 0.9
    num_samples_team_per_call: int = 10
    num_samples_player_per_call: int = 20
    languages: list[str] = ["English", "French"]


class SentenceAnnotationGenerationSettings(BaseSettings):
    # llm_model_id: str = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL"
    llm_model_id: str = "gpt-oss:20b"
    llm_client_url: str = "http://localhost:11434/v1"  # default to ollama
    llm_client_api_key: str = "any"
    temperature: float = 0.0


class SentenceEvaluationSettings(BaseSettings):
    llm_model_id: str = "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
    llm_client_url: str = "http://localhost:11434/v1"  # default to ollama
    llm_client_api_key: str = "any"
    temperature: float = 0.0
