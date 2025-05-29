import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import ModelConfig
from src.tba_trainer_gsm8k import TBAConfigGSM8K, TBATrainerGSM8K
from src.utils import TRLParser, WandbLogModelConfig
from src.gsm8k_utils import prepare_dataset
import torch


from src.dist_utils import init_distributed_env


@dataclass
class ScriptArguments:
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    max_length: int = field(default=256, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    wandb_run_id: Optional[str] = field(default=None)
    use_deepspeed: bool = field(default=False, metadata={"help": "Use DeepSpeed via plugin"})

if __name__ == "__main__":
    # prevent accelerate from joining all processes -- see init_distributed_env for details
    accelerate_ranks = [0, 1]  # Changed from [0] to [0, 1] for multi-GPU training
    
    comm, comm_world_rank, comm_world_size, accelerator = init_distributed_env(
        accelerate_ranks=accelerate_ranks,
        accelerate_kwargs={
        }
    )

    parser = TRLParser((ScriptArguments, TBAConfigGSM8K, ModelConfig))
    print(comm_world_rank, flush=True)
    args, config, model_config = parser.parse_args_and_config()
    config.use_deepspeed = args.use_deepspeed
    
    print(f'Before trainer created, reporting from rank {comm_world_rank} of {comm_world_size}', flush=True)
    
    if args.output_global_parent_dir is not None:
        config.output_dir = os.path.join(args.output_global_parent_dir, config.output_dir)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_path)
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.bfloat16
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.bfloat16
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy.pad_token = tokenizer.eos_token
        ref_policy.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # Dataset
    ################
    assert args.dataset_name == "openai/gsm8k", args.dataset_name
    raw_datasets = load_dataset("openai/gsm8k", "main")

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    print('max dataset length is ', max([x['lengths'] for x in train_dataset]),
          '\nargs.max_length is ', args.max_length, flush=True)
    assert max([x['lengths'] for x in train_dataset])<=args.max_length, max([x['lengths'] for x in train_dataset])
    assert max([x['lengths'] for x in eval_dataset])<=args.max_length, max([x['lengths'] for x in eval_dataset])
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    TrainerCls = TBATrainerGSM8K

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)],
    )
    trainer.train()
                
    if not config.sanity_check and accelerator is not None:
        trainer.save_model(config.output_dir)
