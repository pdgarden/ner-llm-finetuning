# ----------------------------------------------------------------------------------------------------------------------
# Import
from unsloth import FastLanguageModel  # noqa: I001
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import os
from datetime import datetime

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from loguru import logger

from src.constants import ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE, MODELS_DIR
from src.prompt_train import TRAIN_ASSISTANT_PROMPT_NER_RETRIEVAL
from src.type import SplitSyntheticSamplesAnnotated, SyntheticSampleAnnotated

# ----------------------------------------------------------------------------------------------------------------------
# Functions


def convert_sample_to_chatml(sample: SyntheticSampleAnnotated) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TRAIN_ASSISTANT_PROMPT_NER_RETRIEVAL},
        {"role": "user", "content": sample.sentence},
        {"role": "assistant", "content": sample.annotation.model_dump_json()},
    ]


def main() -> None:
    logger.info("Setup model")

    instruct_model = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=instruct_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    logger.info("Load dataset")
    with ANNOTATED_SYNTHETIC_PROCESSED_DATA_FILE.open("r") as f:
        dataset_pyd = SplitSyntheticSamplesAnnotated.model_validate_json(f.read())

    formatting_prompts_func = lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
    }

    dataset = Dataset.from_dict({"conversations": [convert_sample_to_chatml(sample) for sample in dataset_pyd.train]})
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    logger.info("Train")
    trainer.train()

    logger.info("Save")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_model_dir = MODELS_DIR / now
    fine_tune_model_name = "llama3-1_ner_finetune_q4_k_m"
    model.save_pretrained_gguf(current_model_dir / fine_tune_model_name, tokenizer, quantization_method="q4_k_m")

    os.system(  # noqa: S605
        "ollama create llama3-1_ner_finetune:q4_k_m -f"  # noqa: S607
        "/home/pdesj/work/github/ner-llm-finetuning/src/Modelfile"
    )

    logger.info("Done")

# ----------------------------------------------------------------------------------------------------------------------
# Main


if __name__ == "__main__":
    main()
