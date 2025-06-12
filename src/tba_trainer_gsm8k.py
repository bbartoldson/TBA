import gc
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.rloo_trainer import INVALID_LOGPROB
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    get_reward,
    print_rich_table,
    truncate_response,
)
from vllm import LLM, SamplingParams
import wandb

from src.utils import prepare_deepspeed
from src.dist_utils import init_distributed_env, broadcast_weights
from src.dist_data_utils import split_dataset_indices, SubsetByCidDataset, InfIterator
from src.buffer_utils import CommentBuffer, CommentBuffersManager
from src.gsm8k_utils import extract_prediction, evaluate

import re
import random

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)

@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0

@dataclass
class TBAConfigGSM8K(RLOOConfig):
    kl_coef: float = 0.012
    kl_coef_final: float = 0.004
    kl_anneal: bool = True
    kl_coef_decay_stop_iter: int = 500
    kl_coef_decay_target_iter: int = 500
    temperature_sample: bool = False
    top_p_sample: bool = False
    rloo_k: int = 20
    rloo_k_buffer_multiplier: float = 1.2
    initial_buffer_samples: int = 500
    vllm_gpu_memory_utilization: float = 0.2
    sync_interval: int = 2
    on_policy_prob: float = 0.95
    lr_scheduler_type: str = "warmup_stable_decay" 
    warmup_ratio: float = 0.05
    WSD_decay_steps: int = 500
    WSD_stable_steps: int = 450

class TBATrainerGSM8K(Trainer):
    def __init__(
        self,
        config: TBAConfigGSM8K,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy

        ###########################
        #### Distributed Setup ####
        ###########################
        accelerate_ranks = [0, 1] # haven't tested multi-GPU trainers, but should be possible with [0,1] (e.g.)
        accelerate_kwargs={
                            'gradient_accumulation_steps': args.gradient_accumulation_steps,
        }
        if args.use_deepspeed:
            accelerate_kwargs.update( {'deepspeed_plugin': self.create_deepspeed_plugin(len(accelerate_ranks))} )
        self.comm, self.comm_world_rank, self.comm_world_size, self.accelerator = init_distributed_env(
            accelerate_ranks=accelerate_ranks, 
            accelerate_kwargs=accelerate_kwargs
        )
        if self.comm_world_rank == 0:
            print(f"""Trainer configuration:
               - Effective batch size: {config.gradient_accumulation_steps*config.per_device_train_batch_size}
               - VLLM GPU memory utilization (config.vllm_gpu_memory_utilization): {config.vllm_gpu_memory_utilization}
               - bfloat16 : {config.bf16}
               - fp16: {config.fp16}
               - KL coefficient (config.kl_coef): {config.kl_coef}
               - Final KL coefficient (config.kl_coef_final): {config.kl_coef_final}
               - Rollout temperature (config.temperature): {config.temperature} 
               - RLOO K (config.rloo_k): {config.rloo_k}
               - RLOO K buffer multiplier, S/K (config.rloo_k_buffer_multiplier): {config.rloo_k_buffer_multiplier}
               - Initial buffer samples (config.initial_buffer_samples): {config.initial_buffer_samples}
               - Most on policy prob: {config.on_policy_prob}
               - Sync period (config.sync_interval): {config.sync_interval}""", flush=True
            )
        
        self.role = 'trainer' if self.comm_world_rank<=max(accelerate_ranks) else 'searcher'
        
        print(f'In trainer init, {self.role} reporting from rank {self.comm_world_rank} of {self.comm_world_size}',
              flush=True)
        self.n_searchers = self.comm_world_size - len(accelerate_ranks)
        self.n_trainers = len(accelerate_ranks)
        # Create the local subset dataset for this rank, assigning it a subset of comment IDs (CIDs)
        #print('RUNNING WITH LIMITED DATASET IN DEBUG MODE!!!')
        #limit = 10000 # remove this limit to stop debug mode
        if self.role=='trainer':
            my_cids = list(range(len(train_dataset))) # keep as is for multi-gpu as trainer only samples what searcher sends it anyway
        else:
            cid_splits = split_dataset_indices(len(train_dataset), self.n_searchers) #split_dataset_indices(limit, self.n_searchers)
            my_cids = cid_splits[self.comm_world_rank-len(accelerate_ranks)]
        self.comment_buffer_manager = CommentBuffersManager(
            assigned_comment_ids = my_cids, 
            rloo_k = self.args.rloo_k, 
            online_prob=self.args.on_policy_prob,
            max_capacity_per_query = 100,
            inv_temp = 0 # uniform sampling with inverse temp = 0
        )
        train_dataset = SubsetByCidDataset(train_dataset, my_cids)

        ## back to original code ##
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator or DataCollatorWithPadding(tokenizer)
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # set up model and stop token
        #########
        for module in [policy, ref_policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy

        self.num_batches = 0
        
        assert args.per_device_train_batch_size/args.rloo_k==args.per_device_train_batch_size//args.rloo_k
        #########
        ### trainer specifics
        #########
        if self.role == 'trainer':
            #########
            # calculate various batch sizes
            #########
            if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
                args.total_episodes = args.num_train_epochs * self.train_dataset_len
            args.world_size = self.accelerator.num_processes if self.accelerator else 1 # searcher ranks have accelerator=None
            print(f'\n\n****Trainer is running with {self.accelerator.num_processes} accelerator num_processes****\n\n')
            args.local_batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
            )
            args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
            assert args.world_size == len(accelerate_ranks)
            args.batch_size = int(args.local_batch_size * args.world_size)
            print(f'Using effective batch size {args.batch_size}')
            args.mini_batch_size = exact_div(
                args.batch_size,
                args.num_mini_batches,
                "`batch_size` must be a multiple of `num_mini_batches`",
            )
            args.local_mini_batch_size = exact_div(
                args.local_batch_size,
                args.num_mini_batches,
                "`local_batch_size` must be a multiple of `num_mini_batches`",
            )
            if args.whiten_rewards:
                assert (
                    args.local_mini_batch_size >= 8
                ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            # `per_rank_rollout_batch_size` is our `args.local_batch_size`
            # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
            self.num_batches = exact_div(
                args.total_episodes,
                args.batch_size,
                f" total_episodes {args.total_episodes} should be divisible by batch_size {args.batch_size} ",
            )
            self.local_seed = args.seed + self.accelerator.process_index * 100003  # Prime
            if args.num_sample_generations > 0:
                self.sample_generations_freq = max(1, self.num_batches // args.num_sample_generations)
            self.local_dataloader_batch_size = exact_div(
                args.local_batch_size,
                args.rloo_k,
                "`local_batch_size` must be a multiple of rloo_k",
            )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

            if self.args.lr_scheduler_type=='warmup_stable_decay':
                self.args.lr_scheduler_kwargs.update({'num_decay_steps': self.args.WSD_decay_steps,
                                                  'num_stable_steps': self.args.WSD_stable_steps}
                )
            self.create_optimizer_and_scheduler(num_training_steps=self.num_batches)
            
            self.state = OnlineTrainerState(
                is_local_process_zero=self.is_local_process_zero(),
                is_world_process_zero=self.is_world_process_zero(),
            )
    
            default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
            callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
            self.callback_handler = CallbackHandler(
                callbacks,
                self.model,
                self.tokenizer,
                self.optimizer,
                self.lr_scheduler,
            )
            self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
            #if self.comm_world_rank==0:
            self.control = TrainerControl()
    
            self.current_flos = 0
            self.hp_search_backend = None
            self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
            self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
            # Create distant repo and output directory if needed
            self.hub_model_id = None
            if self.args.push_to_hub:
                self.init_hf_repo()
            if self.args.should_save:
                os.makedirs(self.args.output_dir, exist_ok=True)
            self.backup_model = None
    
            # the trainer will just pull training data from the buffer
            self.dataloader = None
            print('Enable Gradient Checkpointing if your batch does not fit', flush=True)
            #if args.per_device_train_batch_size>20:
            #    self.model.gradient_checkpointing_enable()
            # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
            # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
            torch.manual_seed(args.seed)
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            torch.manual_seed(self.local_seed)  # reset the local seed again
    
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=DataCollatorWithPadding(self.tokenizer),
                drop_last=False,
                shuffle=False,
            )  # no need to shuffle eval dataset
            #self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
    
            del self.ref_policy # not used by the trainer
                
        args.num_updates = self.num_batches * args.num_mini_batches
        args.batch_size = self.comm.bcast(args.batch_size, root=0)
        self.num_batches = self.comm.bcast(self.num_batches, root=0)
        # how many samples per inference iter
        self.n_repeats = int(args.rloo_k_buffer_multiplier * args.rloo_k)
        if self.role=='searcher':
            self.model = self.model.cuda()
            self.ref_policy = self.ref_policy.cuda()

            #################
            ## set up vLLM ##
            #################
            if args.fp16:
                vllm_dtype = torch.float16
            elif args.bf16:
                vllm_dtype = torch.bfloat16
            else:
                vllm_dtype = torch.float32
                
            self.generation_config, self.llm = self.load_vllm(
                args.sft_model_path,
                args.vllm_gpu_memory_utilization,
                vllm_dtype,
                args.temperature,
                args.response_length
            )
        
            if self.n_repeats>args.local_rollout_forward_batch_size:
                searcher_bs = 1
            else:
                searcher_bs = math.ceil(args.local_rollout_forward_batch_size/self.n_repeats)
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size= searcher_bs, 
                shuffle= False,
                collate_fn= DataCollatorWithPadding(tokenizer),
                drop_last= False,  # needed; otherwise the last batch will be of ragged shape
            )
            
        self.trainer_iteration = 0
        self.synced_iterations = {0}
        self.changed_cids = set()
        self.sync_interval = sync_interval = args.sync_interval
        self.max_sync_iteration = sync_interval * ((self.num_batches  + sync_interval - 1) // sync_interval - 1)
        self.initial_buffer_samples = args.initial_buffer_samples
        self.args.kl_coef_original = self.args.kl_coef
        #TODO, we shouldn't have to set these manually. might need to pass deepspeed to parser
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L2012
        self.model_wrapped=self.deepspeed=self.model

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader
    
    def grade_answer(self, given_answer, ground_truth):
        comparisons = (given_answer == ground_truth).float() 
        return comparisons
        
    def extract_predicted_answers(self, texts: List[str]) -> List[Optional[str]]:
        """
        Extract predicted answers from a list of texts.
        """
        answers = []
        for text in texts:
            answer = extract_prediction(text)
            answers.append(answer)
        return answers
    
    def evaluate(self):
        self.model.eval()
        
        for temp in [0]:
            if temp==0:
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.response_length,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id]
                )
            else:
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.response_length,
                    temperature=temp, 
                    do_sample=True,
                    eos_token_id=[self.tokenizer.eos_token_id]
                )

            with torch.no_grad():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    correct, total = evaluate(
                        unwrapped_model, 
                        self.eval_dataloader, 
                        generation_config, 
                        self.tokenizer
                    )
                    accuracy = correct / total * 100
                    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

            wandb.log({f'test/final accuracy (temp={temp})': correct / total})
        self.model.train()

    def train(self):
        
        self.init_buffer(self.n_repeats)
        self.comm.barrier()
        self.sync(data_only=True)
        print(f'{self.role}, rank {self.comm_world_rank}, synced on trainer iter',
                  self.trainer_iteration, flush=True
        )
        
        if self.role == 'trainer':
            self.generate_completions(sampling=True, init=True)

            self.trainer_loop()

            self.evaluate()
        else:
            self.searcher_loop()
            
        self.comm.barrier()
        print('All processes finished training. Exiting trainer.')
        
    def searcher_loop(self):
        # do search iters to add to the local copy of the comment buffer manager
        # broadcast results to trainer's comment buffer manager at sync steps

        inf_iter = InfIterator(self.dataloader)
        
        while self.trainer_iteration < self.max_sync_iteration:
            # 1) Drain iteration messages
            while self.comm.Iprobe(source=0):
                new_val = self.comm.recv(source=0)
                self.trainer_iteration = max(self.trainer_iteration, new_val)
            
            if (self.trainer_iteration > 0 and
                self.trainer_iteration % self.sync_interval == 0 and
                self.trainer_iteration not in self.synced_iterations
            ):
                self.sync()
                print(f'{self.role}, rank {self.comm_world_rank}, synced on trainer iter',
                    self.trainer_iteration, flush=True
                )
            
            batch_of_data = next(inf_iter)
            new_items = self.search_iter(batch_of_data, self.n_repeats)
            self.add_to_comment_buffers(new_items)
        
    def search_iter(self, batch_of_data, n_repeats):

        args = self.args
        model = self.model
        ref_policy = self.ref_policy
        tokenizer = self.tokenizer

        model.train()
        ref_policy.train()
        
        ##########################
        #### vLLM generation #####
        ##########################
        cids = batch_of_data["cid"].repeat(n_repeats).numpy()
        queries = batch_of_data["input_ids"].cuda()
        # TODO: create an arg that sets length of queries
        queries = queries.repeat(n_repeats, 1) 
        response_d = batch_of_data["response_ids"].cuda()
        response_d = response_d.unsqueeze(1)  
        response_d = response_d.repeat(n_repeats, 1) 
        context_length = queries.shape[1]
        query_responses = []
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        accuracies = []
        sequence_lengths = []
        
        g_queries_list = queries.tolist()
        g_queries_list = [
            [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id] for item in g_queries_list
        ]  # remove padding
        
        # Off-policy sampling
        top_p = 1.0
        temp = args.temperature + 1e-7
        if args.top_p_sample and np.random.binomial(1, 0.3):
            top_p = np.random.uniform(0.7, 1.0)
        if args.temperature_sample and np.random.binomial(1, 0.3):
            temp = np.random.uniform(temp, temp*(1.1/0.7)) + 1e-7
        generation_config = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        kl_coef_start = args.kl_coef_original
        if args.kl_anneal:
            if self.trainer_iteration < args.kl_coef_decay_stop_iter:
                args.kl_coef = args.kl_coef_final * (self.trainer_iteration-1)/args.kl_coef_decay_target_iter + kl_coef_start * (1 - (self.trainer_iteration-1)/args.kl_coef_decay_target_iter)
            else:
                args.kl_coef = args.kl_coef_final

        with torch.no_grad():
            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]
                response_d_mini = response_d[i : i + args.local_rollout_forward_batch_size]
                g_query = g_queries_list[i : i + args.local_rollout_forward_batch_size]
                vllm_response = self.get_vllm_responses(
                    g_query,
                    generation_config
                ).cuda()
                query_response = torch.cat((query, vllm_response), 1)
                response = query_response[:, context_length:]
                
                output = forward(model, query_response, tokenizer.pad_token_id)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.temperature + 1e-7
                # use the logits during generation directly, instead of using the following
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()

                ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        args.stop_token_id, tokenizer.pad_token_id, response
                    )
                    
                # pred_answer = self.extract_predicted_answers(decoded_text, use_original_format=False)
                pred_answer = tokenizer.batch_decode(response)
                pred_answer = self.extract_predicted_answers(pred_answer)
                #response_d
                response_value =  torch.tensor(pred_answer).view(len(pred_answer), 1).cuda()
                score = (self.grade_answer(response_value, response_d_mini)).squeeze(1) # binary_RM

                correct_predictions = score.sum()
                # Calculate accuracy using the sum of correct predictions divided by the total number of predictions
                total_predictions = len(score)  # or ground_truth.size(0) for number of rows
                accuracy = correct_predictions / total_predictions

                # Response Processing 2. run reward model on the truncated responses
                #postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                #_, score, _ = get_reward(
                #    postprocessed_query_response, tokenizer.pad_token_id, context_length
                #)

                query_responses.append(query_response)
                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                accuracies.append(accuracy)
            query_responses = torch.cat(query_responses, 0)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            accuracies = torch.tensor(accuracies)
            #print("===Accuracy:", accuracies.mean())
            
            del (logprob, ref_logprob, score, accuracies)
            torch.cuda.empty_cache()
            gc.collect()

            # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
            # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            non_score_reward = (-args.kl_coef * kl).sum(1)
            rlhf_reward = scores + non_score_reward

            # vectorized RLOO advantages implementation
            rlhf_reward = rlhf_reward.reshape(n_repeats, -1)
            baseline = (rlhf_reward.sum(0) - rlhf_reward) / (n_repeats - 1)
            advantages = rlhf_reward - baseline
            advantages = advantages.flatten()
            
            responses = responses.cpu().numpy()
            advantages = advantages.cpu().numpy() # TODO, print some of these?
            scores = scores.cpu().numpy()
            logprobs = logprobs.cpu().numpy()
            ref_logprobs = ref_logprobs.cpu().numpy()
            sequence_lengths = sequence_lengths.cpu().numpy()
            
            torch.cuda.empty_cache()
            gc.collect()

        items = [{
            "policys_trainer_iteration": self.trainer_iteration,
            'cid': cids[i],
            'response': responses[i],
            'advantage': advantages[i],
            'score': scores[i],
            'logprob': logprobs[i],
            'ref_logprob': ref_logprobs[i],
            'sequence_length': sequence_lengths[i] # allows you to recompute padding mask
        } for i in range(len(advantages))]
        return items

    def trainer_loop(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        tokenizer = self.tokenizer
        device = accelerator.device

        accelerator.print("===training policy===")
        self.state.global_step = 0
        self.state.episode = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()
        kl_coef_start = args.kl_coef
        logZ_k_size = args.rloo_k
        self.state.max_steps = args.num_updates
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        if self.accelerator.process_index == 0:
            wandb.log(self.init_table)
        for update in range(1, self.num_batches + 1):
            if args.kl_anneal:
                if update < args.kl_coef_decay_stop_iter:
                    args.kl_coef = args.kl_coef_final * (update-1)/args.kl_coef_decay_target_iter + kl_coef_start * (1 - (update-1)/args.kl_coef_decay_target_iter)
                else:
                    args.kl_coef = args.kl_coef_final
            self.trainer_iteration = update
            self.state.episode += 1 * args.batch_size
            self.lr_scheduler.step()
                
            if self.trainer_iteration<=self.max_sync_iteration:
                if self.comm_world_rank == 0:
                    for w in range(self.comm_world_size-self.n_searchers, self.comm_world_size):
                        self.comm.isend(self.trainer_iteration, dest=w)
            
            (responses, sequence_lengths, advantages, logprobs,
             ref_logprobs, scores, query_responses, context_length,
             padding_mask, kl, non_score_reward, rlhf_reward) = self.get_batch_from_buffer(args.local_batch_size)
            
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.arange(args.local_batch_size)#np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                            # TB
                            p_ref_f = ref_logprobs.sum(1)
                            pi_f = new_logprobs.sum(1)
                            log_Z_pred = ((-pi_f + p_ref_f[micro_batch_inds]) + scores[micro_batch_inds]/args.kl_coef).view(args.rloo_k, -1)[:logZ_k_size].mean(0).repeat(args.rloo_k).detach()
                            tb_loss = ((log_Z_pred + (pi_f - p_ref_f[micro_batch_inds]) - scores[micro_batch_start:micro_batch_end]/args.kl_coef)**2).mean()
                            accelerator.backward(tb_loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            #loss = pg_loss
                            #accelerator.backward(loss)
                            #optimizer.step()
                            #optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1
                    #time.sleep(12)
                    #if self.comm_world_rank==0:
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.control.should_save:
                        print(f'rank {self.comm_world_rank} is SAVING!!!')
                        self._save_checkpoint(model, trial=None, metrics=None)
                        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["loss/logZ"] = self.accelerator.gather(log_Z_pred).mean().item()
                metrics["loss/tb_loss"] = self.accelerator.gather(tb_loss).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["val/kl_coef"] = args.kl_coef
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                if self.comm_world_rank==0:
                    self.log(metrics)
            del kl, mean_kl, mean_entropy, scores
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update) % self.sample_generations_freq == 0:
                self.generate_completions()

            if (self.trainer_iteration % self.sync_interval == 0 and
                    self.trainer_iteration < self.num_batches):
                self.sync()
                print(f'{self.role}, rank {self.comm_world_rank}, synced on trainer iter',
                    self.trainer_iteration, flush=True
                )

        #if self.comm_world_rank==0:
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            print(f'rank {self.comm_world_rank} is SAVING!!!')
            self._save_checkpoint(model, trial=None, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False, init=False):
        '''
        self.init_table = {"completion_table": None}
        return
        '''
        self.model.eval()
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id]
        )

        scores = []
        table = defaultdict(list)

        for i, batch in enumerate(self.eval_dataloader):
            print(f'rank {self.comm_world_rank} has batch {i} with {len(batch["input_ids"])} elements',flush=True)
            
            query = batch["input_ids"].cuda()
            response_d = batch["response_ids"].cuda()
            ground_truth = response_d.unsqueeze(1)
        
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = generate(
                        unwrapped_model,
                        query,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                response = query_response[:, context_length:]
                                
                pred_answer = tokenizer.batch_decode(response)
                pred_answer = self.extract_predicted_answers(pred_answer)
                response_value =  torch.tensor(pred_answer).view(len(pred_answer), 1).to(self.accelerator.device)
                score = (self.grade_answer(response_value, ground_truth)).squeeze(1) # binary_RM
                scores.append(score.sum().cpu().item())
                if i == 0:
                    table["query"].extend(tokenizer.batch_decode(query))
                    table["model response"].extend(tokenizer.batch_decode(response))
                    table["model response value"].extend(response_value.squeeze().float().cpu().numpy())
                    table["test acc approx"].extend(score.float().cpu().numpy())
                
        
        if self.comm_world_rank==0:
            df = pd.DataFrame(table)
            print_rich_table(df.iloc[0 : 0 + 5])
            if init:
                self.init_table = {"completion_table": wandb.Table(dataframe=df)}
            else:
                wandb.log({"completion_table": wandb.Table(dataframe=df)})
                wandb.log({"test/one batch acc": sum(scores)/len(self.eval_dataloader.dataset)})


    def get_vllm_responses(self, g_queries_list, generation_config_override=None):
        g_response_ids = self.vllm_generate(g_queries_list, generation_config_override)

        DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
        g_padded_response_ids = [
            list(response) + [DUMMY_PAD_TOKEN] * (self.args.response_length - len(response))
            for response in g_response_ids
        ]
        vllm_responses = torch.tensor(g_padded_response_ids)
        
        return vllm_responses

    def vllm_generate(self, g_queries_list, generation_config_override=None):
        
        outputs = self.llm.generate(
            prompt_token_ids=g_queries_list,
            sampling_params=generation_config_override or self.generation_config,
            use_tqdm=False
        )
        response_token_ids = []
        for output in outputs:
            response_token_ids.append(output.outputs[0].token_ids)

        return response_token_ids

    def load_vllm(self, 
                  model_name_or_path,
                  vllm_gpu_memory_utilization,
                  vllm_dtype,
                  temperature,
                  response_length
        ):
        
        generation_config = SamplingParams(
            temperature=(temperature + 1e-7),
            top_p=1.0,
            max_tokens=response_length,
            include_stop_str_in_output=True,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
      
        llm = LLM(
            model=model_name_or_path,
            revision="main",
            tokenizer_revision="main",
            tensor_parallel_size=1,
            #device=vllm_device,
            dtype=vllm_dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        print(f"🔥🔥🔥 vllm loaded in {vllm_dtype}")

        return generation_config, llm

    def init_buffer(self, n_repeats):
        # init the buffer with n_repeats responses per query in the local manager

        # Calculate number of sync points (one every 10 batches)
        total_batches = len(self.dataloader) if self.role=='searcher' else 1e10
        n_sync_points = total_batches// 100
        all_sync_points = self.comm.gather(n_sync_points, root=0)
        if self.comm_world_rank == 0: 
            n_sync_points = min(all_sync_points)
            n_sync_points = max(n_sync_points,1)
        n_sync_points = self.comm.bcast(n_sync_points, root=0)
        self.comm.barrier()

        syncs = 0
        examples_in_buffer = 0
        start = time.time()
        if self.role=='searcher':
            for i, batch_of_data in enumerate(self.dataloader):
                new_items = self.search_iter(batch_of_data, n_repeats)
                self.add_to_comment_buffers(new_items)
                examples_in_buffer += self.args.local_rollout_forward_batch_size*self.n_searchers
                if i%100 == 0 and i>0 and syncs<n_sync_points-1: # send data in super-batches
                    self.sync(data_only=True)
                    syncs+=1
                    if self.comm_world_rank == self.comm_world_size - self.n_searchers:
                        print(f'\nAfter {time.time()-start} seconds, there are {examples_in_buffer}',
                              'examples in initial buffer.', flush=True)
                if examples_in_buffer > self.initial_buffer_samples:
                    break
            if self.comm_world_rank == self.comm_world_size - self.n_searchers:
                print(f'\nAfter {time.time()-start} seconds, there are {examples_in_buffer}',
                      'examples in initial buffer.', flush=True)
        while syncs<n_sync_points:
            self.sync(data_only=True)
            syncs+=1
        
    def add_to_comment_buffers(self, new_items):
        updates_by_cid = {}
        for item in new_items:
            if item['cid'] in updates_by_cid:
                updates_by_cid[item['cid']].append(item)
            else:
                updates_by_cid[item['cid']] = [item]
        for cid, item_list in updates_by_cid.items():
            self.comment_buffer_manager.comment_buffers[cid].add_new_items(
                item_list, 
                self.trainer_iteration
            )
            self.changed_cids.add(cid)

    def sync(self, data_only=False):
        self.comm.barrier()
        
        if not data_only:
            self.sync_weights()
            
        if self.role=='trainer':
            # Gather from workers
            for corresponding_searcher in range(self.n_trainers+self.comm_world_rank, self.comm_world_size, self.n_trainers):
                searcher_dict = self.comm.recv(source=corresponding_searcher)
                for cid in searcher_dict:
                    print(f'Trainer rank {self.comm_world_rank} received CID {cid} on iter {self.trainer_iteration}', flush=True)
                    self.comment_buffer_manager.overwrite_cid_buffer(
                        cid,
                        searcher_dict[cid],
                        self.trainer_iteration
                    )
        else:
            corresponding_trainer = (self.comm_world_rank-self.n_trainers) % self.n_trainers
            updated_data = {}
            for cid in self.changed_cids:
                print(f'Searcher rank {self.comm_world_rank} sent CID {cid} on iter {self.trainer_iteration}', flush=True)
                # We send the entire CommentBuffer object
                updated_data[cid] = self.comment_buffer_manager.comment_buffers[cid]
            self.comm.send(updated_data, dest=corresponding_trainer)
            self.synced_iterations.add(self.trainer_iteration)
            self.changed_cids = set()

    def sync_weights(self):
        self.comm.barrier()
        start = time.time()
        if self.role=='trainer':
            broadcast_weights(self.model, self.comm, root_mpi_rank=0, role=self.role)
            print(f"broadcast weights took: {time.time() - start:.2f} seconds", flush=True)
        else:
            broadcast_weights(self.model, self.comm, root_mpi_rank=0, role=self.role)
            llmp = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llmp.load_weights(self.model.named_parameters())
            print('Loaded updated parameters into vLLM engine')

    def get_batch_from_buffer(self, batch_size):
        ########### Distributed Async Buffer Usage #############
        args = self.args
        
        # get and unpack buffer data
        #start_data = time.time()
        buffer_data, query_IDs = self.comment_buffer_manager.get_batch(batch_size)
        responses = buffer_data['responses'].to(self.accelerator.device)
        advantages = buffer_data['advantages'].to(self.accelerator.device)
        scores = buffer_data['scores'].to(self.accelerator.device)
        logprobs = buffer_data['logprobs'].to(self.accelerator.device)
        ref_logprobs = buffer_data['ref_logprobs'].to(self.accelerator.device)
        sequence_lengths = buffer_data['sequence_lengths'].to(self.accelerator.device)
        
        # get full, padded queries from their keys
        queries = self.data_collator([self.train_dataset[i] for i in query_IDs])['input_ids'].to(self.accelerator.device)
        queries = queries.repeat_interleave(args.rloo_k, dim=0, output_size=args.rloo_k*len(query_IDs))
        context_length = queries.shape[1]
        
        # recompute padding mask and query_responses
        query_responses = torch.cat((queries, responses), 1)
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

        # recompute stats
        kl = logprobs - ref_logprobs
        non_score_reward = (-args.kl_coef * kl).sum(1)
        rlhf_reward = scores + non_score_reward
        #print(f'Produced buffer data batch in {time.time()-start_data:.2f} seconds', flush=True)

        return (
                    responses, sequence_lengths, advantages, logprobs, ref_logprobs,
                    scores, query_responses, context_length, padding_mask,
                    kl, non_score_reward, rlhf_reward
               )

    def create_deepspeed_plugin(self, n_trainers):
        args = self.args

        from accelerate.utils import DeepSpeedPlugin
        from accelerate.utils.deepspeed import HfDeepSpeedConfig
        config = {
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "zero_optimization": {"stage": 2},
                "bf16.enabled": True,
                "fp16.enabled": False,
                "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
                "train_batch_size": args.per_device_train_batch_size * 
                                    args.gradient_accumulation_steps *
                                    n_trainers
        }
        config = HfDeepSpeedConfig(config)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config)
        '''
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=1.0
        )
        kwargs = {
            "bf16.enabled": True,
            "fp16.enabled": False,
            "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
            "train_batch_size": args.per_device_train_batch_size * 
                                args.gradient_accumulation_steps *
                                n_trainers
        }
        deepspeed_plugin.deepspeed_config_process(**kwargs)
        '''
        print(deepspeed_plugin, flush=True)

        
        return deepspeed_plugin
