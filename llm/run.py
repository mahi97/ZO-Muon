import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from HiZOOtrainer import HiZOOTrainer
from MeZOTrainer import MeZOTrainer
from LowDimTrainer import LowDimTrainer
from LOZOTrainer import LOZOTrainer
from SubZeroTrainer import SubZeroTrainer
from KFACMeZOTrainer import KFACMeZOTrainer
from ZOAdamTrainer import ZOAdamTrainer
from LORENTrainer import LORENTrainer
from transformers import (
    TrainerCallback, 
    TrainerControl, 
    TrainerState, 
)
import random
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    trainer: str= "hessian"
    # Number of examples
    num_train: int = 0 # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None # (only enabled with training) number of development samples
    num_eval: int = None # number of evaluation samples
    num_train_sets: int = None # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = None # designated seed to sample training samples/demos
    result_file: str = None # file name for saving performance; if None, then use the task name, model name, and config
    do_grad_scaling: bool = False

    # Model loading
    model_name: str = "facebook/opt-125m" # HuggingFace model name
    load_float16: bool = False # load model parameters as float16
    load_bfloat16: bool = False # load model parameters as bfloat16
    load_int8: bool = False # load model parameters as int8
    max_length: int = 2048 # max length the model can take
    no_auto_device: bool = False # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples

    zo_optimizer: str = "sgd"  # "sgd" or "adam"
    multiple_sample: bool=False
    max_time: int=0
    num_samples: int=4
    step_interval: int = 50 
    rank_r: int = 2 
    sparsity: float=1.0
    # [inverse-metric] Flags passed by the released subzero.sh / lowdim.sh that were not defined here
    # (the scripts crashed in HfArgumentParser). Accepted and IGNORED: no trainer reads them.
    reduce_rank: bool = False
    truncation_rank: int = 2
    switch_optim: bool = False
    threshold: int = 15000
    compute_variance: bool = False

    # [inverse-metric] K-FAC-shaped MeZO (KFACMeZOTrainer, trainer=kfac_mezo)
    kfac_placement: str = "A"  # "A": perturb with E = c*G^p Z A^p (cov = c^2 K^{2p}); "B": isotropic perturb, update with c^2 G^{2p} (g z) A^{2p}
    kfac_power: float = -0.5  # p. -0.5 => cov K-FAC^-1; -0.25 => K-FAC^-1/2; +0.5 => K-FAC^+1; 0 => isotropic (bit-identical MeZO)
    kfac_damping: float = 1e-3  # damping gamma, relative: lambda += gamma * mean(lambda) per factor
    kfac_block: int = 512  # block-diagonal block size for A and G
    kfac_calib_steps: int = 200  # backprop minibatches per calibration
    kfac_calib_bs: int = 16  # examples per calibration minibatch
    kfac_calib_microbs: int = 16  # at most this many examples per backward pass
    kfac_calib_max_tokens: int = 1200  # at most this many padded tokens per backward pass (one long example may exceed it)
    kfac_fisher: str = "true"  # "true": labels sampled from the model's predictive distribution; "empirical": gold labels
    kfac_refresh: int = 2000  # recompute factors every N ZO steps (0 = never)
    kfac_cache_E: bool = True  # materialize E once per step (+1x param memory) instead of regenerating 4x
    kfac_spectra_file: str = None  # where to write the factor spectra JSON (first calibration); default <output_dir>/kfac_spectra.json
    kfac_eigh_fp64: bool = True  # eigendecompose blocks in fp64
    kfac_scope: str = "all"  # which parameters are perturbed and updated: all | weights (K-FAC'd weights only) | rest (embeddings, positions, 1-D only)
    kfac_pg_decomp: int = 0  # if >0: on the first N steps split the directional derivative by parameter subset (diagnostic, 6 extra forwards)
    kfac_check_restore: int = 0  # if >0: on the first N steps record max|W_after_restore - W_before| (diagnostic)

    # [inverse-metric] LOREN port (LORENTrainer, trainer=loren); defaults from the released README/main.py
    loren_num_queries: int = 10  # K; main.py default (the README leaves it unset for LOREN)
    loren_damping: float = 0.1
    loren_lr_cov: float = 1e-3
    loren_weight_decay: float = 0.01  # effective value in the released code, see LORENTrainer.py

    # [inverse-metric] apply lm_head only to the option positions in the training/eval loss (memory; same loss up to rounding)
    slice_logits: bool = False

    # [inverse-metric] periodic accuracy eval (dev + test) every N steps, logged to W&B (0 = off)
    acc_eval_steps: int = 0
    # Training
    trainer: str = "none" 
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order (MeZO) training
    only_train_option: bool = True # whether to only train the option part of the input
    train_as_classification: bool = False # take the log likelihood of all options and train as classification 

    # MeZO
    zo_eps: float = 1e-3 # eps in MeZO
    
    ###############diag
    warmup_step: int = 0
    decay_step: int = 0
    zo_lr_scheduler_type: str = 'constant'
    weight_decay: float = 0
    hessian_smooth_type: str = 'constant0'

    # Prefix tuning
    prefix_tuning: bool = False # whether to use prefix tuning
    num_prefix: int = 5 # number of prefixes to use
    no_reparam: bool = True # do not use reparameterization trick
    prefix_init_by_real_act: bool = True # initialize prefix by real activations of random words

    # LoRA
    lora: bool = False # whether to use LoRA
    lora_alpha: int = 16 # alpha in LoRA
    lora_r: int = 8 # r in LoRA

    # Generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # Saving
    save_model: bool = False # whether to save the model
    no_eval: bool = False # whether to skip evaluation
    tag: str = "" # saving tag

    # Linear probing
    linear_probing: bool = False # whether to do linear probing
    lp_early_stopping: bool = False # whether to do early stopping in linear probing
    head_tuning: bool = False # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False # untie the embeddings and LM head

    # Display
    verbose: bool = False # verbose output

    # Non-diff objective
    non_diff: bool = False # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False # save model when interrupted (useful for long training)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MaxTimeCallback(TrainerCallback):
    """
    A callback that stops training after a specified number of seconds.
    """
    def __init__(self, max_seconds):
        self.start_time = time.time()
        self.max_seconds = max_seconds

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\n[MaxTimeCallback] Time limit of {self.max_seconds}s reached. Stopping training.")
            control.should_training_stop = True # <--- THIS WAS MISSING OR IMPLIED
            return control



class StepTimerCallback(TrainerCallback):
    """[inverse-metric] wall-clock per ZO step (CUDA-synchronized) and peak memory."""
    def __init__(self, skip=5):
        self.skip = skip
        self.times = []
        self._t = None
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self._t = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if self._t is not None:
            dt = time.time() - self._t
            tr = getattr(self, "trainer", None)  # exclude K-FAC calibration time (reported separately)
            if tr is not None and getattr(tr, "_calib_seconds_pending", 0.0) > 0:
                dt -= tr._calib_seconds_pending
                tr._calib_seconds_pending = 0.0
            self.times.append(dt)
        if state.global_step % 100 == 0 and len(self.times) > self.skip:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"perf/s_per_step": float(np.mean(self.times[-100:])),
                               "perf/peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30,
                               "train/global_step": state.global_step}, step=state.global_step)
            except Exception:
                pass
    def summary(self):
        t = np.array(self.times[self.skip:]) if len(self.times) > self.skip else np.array(self.times)
        out = {"n_steps_timed": int(len(t))}
        if len(t):
            out.update({"s_per_step_mean": float(t.mean()), "s_per_step_median": float(np.median(t))})
        out["peak_mem_allocated_GB"] = torch.cuda.max_memory_allocated() / 2**30
        out["peak_mem_reserved_GB"] = torch.cuda.max_memory_reserved() / 2**30
        return out


class AccEvalCallback(TrainerCallback):
    """[inverse-metric] dev/test accuracy every N steps (same evaluate() as the final eval), logged to W&B."""
    def __init__(self, framework, dev_samples, test_samples, every):
        self.fw, self.dev, self.test, self.every = framework, dev_samples, test_samples, every
        self.history = []
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0 or state.global_step == state.max_steps:
            return
        rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), torch.cuda.get_rng_state_all())
        rec = {"step": state.global_step}
        if self.dev is not None:
            rec["dev_accuracy"] = self.fw.evaluate([], self.dev).get("accuracy")
        if self.test is not None:
            rec["test_accuracy"] = self.fw.evaluate([], self.test).get("accuracy")
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.random.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        self.history.append(rec)
        logger.info(f"[acc_eval] {rec}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/dev_acc": rec.get("dev_accuracy"), "eval/test_acc": rec.get("test_accuracy"),
                           "train/global_step": state.global_step}, step=state.global_step)
        except Exception:
            pass


from transformers import Trainer

class RegularTrainer(Trainer):
    """
    A custom Trainer that fixes the compatibility issue between standard AdamW 
    and the modified transformers library used in ZO/MeZO environments.
    """
    def create_optimizer(self):
        # Create the standard optimizer (AdamW) using the parent class logic
        optimizer = super().create_optimizer()
        
        # Monkey-patch: Add a dummy 'train' method if it doesn't exist.
        if not hasattr(optimizer, "train"):
            optimizer.train = lambda: None

        # Monkey-patch: Add a dummy 'eval' method if it doesn't exist.
        # This prevents the AttributeError: 'AdamW' object has no attribute 'eval'
        if not hasattr(optimizer, "eval"):
            optimizer.eval = lambda: None
            
        return optimizer

class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name, trust_remote_code=True)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                # Head tuning
                from ht_opt import OPTForCausalLM
                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    max_memory={i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())},
                    load_in_8bit=self.args.load_int8, trust_remote_code=True
                )
                
                
            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        tokenizer.pad_token_id = 0
        
        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer

    

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]


    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics


    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))
            self.model._slice_logits = self.args.slice_logits  # [inverse-metric]
        #########################################

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification
        if self.args.trainer=="hizoo":
            trainer = HiZOOTrainer(
                    model=self.model, 
                    args=self.args,
                    train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
                )
        elif self.args.trainer=="mezo":
            trainer = MeZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer=="lowdim":
            trainer = LowDimTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer=="lozo":
            trainer = LOZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer in ("zo_adam", "loren"):
            trainer = {"zo_adam": ZOAdamTrainer, "loren": LORENTrainer}[self.args.trainer](
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer=="kfac_mezo":
            trainer = KFACMeZOTrainer(
                model=self.model,
                args=self.args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer=="subzero":
            trainer = SubZeroTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        else:
            # Use the custom RegularTrainer here
            trainer = RegularTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # [inverse-metric] The ZO trainers call self._save_checkpoint directly at every eval (best_eval_loss is
        # never updated), writing a full checkpoint each time. With --save_strategy no we disable that.
        if str(self.args.save_strategy) in ("no", "IntervalStrategy.NO", "SaveStrategy.NO"):
            trainer._save_checkpoint = lambda *a, **k: None
            logger.info("[inverse-metric] save_strategy=no: checkpoint saving disabled")

        # [inverse-metric] step timing + peak memory, and optional periodic accuracy eval
        self.step_timer = StepTimerCallback()
        self.step_timer.trainer = trainer
        trainer.add_callback(self.step_timer)
        if self.args.acc_eval_steps > 0:
            trainer.add_callback(AccEvalCallback(self, eval_samples, getattr(self, "test_samples", None), self.args.acc_eval_steps))
        self.trainer = trainer
        
        # ==========================================
        # NEW: Inject MaxTimeCallback if requested
        # ==========================================
        if hasattr(self.args, "max_time") and self.args.max_time is not None and self.args.max_time > 0:
            logger.info(f"Enabling MaxTimeCallback: Training will stop after {self.args.max_time} seconds.")
            trainer.add_callback(MaxTimeCallback(max_seconds=self.args.max_time))
        
        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # Train loop
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint) 
        
        # Log completion
        logger.info(f"Training finished. Global step: {train_result.global_step}")

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


# def main():
#     args = parse_args()

#     set_seed(args.seed)
#     task = get_task(args.task_name)
#     train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

#     # Initialize trainer and load model
#     framework = Framework(args, task)
#     if args.train_set_seed is not None or args.num_train_sets is not None:
#         # Eval samples share one (or multiple) training set(s)
#         for train_set_id, train_samples in enumerate(train_sets):
#             train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

#             # Sample eval samples
#             if args.num_eval is not None:
#                 eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
#             else:
#                 eval_samples = task.valid_samples

#             if args.trainer != "none":
#                 if args.num_dev is not None:
#                     # Dev samples
#                     dev_samples = train_samples[-args.num_dev:] 
#                     train_samples = train_samples[:-args.num_dev]
#                 else:
#                     dev_samples = None

#                 # Training
#                 framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)

#                 if not args.no_eval:
#                     metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
#                     if dev_samples is not None:
#                         dev_metrics = framework.evaluate([], dev_samples) 
#                         for m in dev_metrics:
#                             metrics["dev_" + m] = dev_metrics[m]
#             else:
#                 assert args.num_dev is None
#                 # Zero-shot / in-context learning
#                 metrics = framework.evaluate(train_samples, eval_samples)

#             if not args.no_eval:
#                 logger.info("===== Train set %d =====" % train_set_seed)
#                 logger.info(metrics)
#                 if args.local_rank <= 0:
#                     write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

#     else:
#         # For each eval sample, there is a training set. no training is allowed
#         # This is for in-context learning (ICL)
#         assert args.trainer == "none"
#         if args.num_eval is not None:
#             eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
#         else:
#             eval_samples = task.valid_samples

#         metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
#         logger.info(metrics)

# if __name__ == "__main__": 
#     main()

import time
import logging

# Assuming other necessary imports (like parse_args, set_seed, etc.) are already present in your file.
# If logger is not defined globally, you might need to initialize it. 
# For this snippet, I assume 'logger' is available as per your original code.

def log_final_to_wandb(metrics, args):
    """[inverse-metric] final test/dev accuracy -> W&B (run summary + one log row). No-op if W&B is off."""
    if "wandb" not in str(args.report_to):
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "inverse-metric-es"), name=args.run_name, config=metrics.get("config"))
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat["final/" + ("test_" + k if k == "accuracy" or k == "f1" else k)] = v
    wandb.log(flat)
    for k, v in flat.items():
        wandb.run.summary[k] = v
    wandb.config.update(metrics.get("config", {}), allow_val_change=True)
    wandb.finish()


def main():
    args = parse_args()

    # [inverse-metric] hard per-process GPU memory cap (allocator only; the CUDA context adds ~0.3-0.5 GB) for shared GPUs
    cap_gb = float(os.environ.get("P3A_GPU_MEM_CAP_GB", 0))
    if cap_gb > 0:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(cap_gb / total_gb, 0)
        logger.info(f"[inverse-metric] GPU memory cap {cap_gb} GB of {total_gb:.1f} GB")

    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    # --- Start Timer ---
    total_start_time = time.time()
    
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                logger.info(f"Starting training for set {train_set_id}...")
                framework.test_samples = eval_samples  # [inverse-metric] for the optional periodic acc eval
                train_start_time = time.time()
                
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
                
                train_end_time = time.time()
                train_duration = train_end_time - train_start_time
                logger.info(f"Training for set {train_set_id} finished in {train_duration:.2f} seconds.")

                if not args.no_eval:
                    _t_eval = time.time()
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
                    metrics["final_eval_seconds"] = time.time() - _t_eval
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                # [inverse-metric] provenance, timing, memory, method-specific extras (e.g. K-FAC backprop count)
                if args.trainer != "none":
                    metrics["train_seconds"] = train_duration
                    if getattr(framework, "step_timer", None) is not None:
                        metrics.update({"perf_" + k: v for k, v in framework.step_timer.summary().items()})
                    tr = getattr(framework, "trainer", None)
                    with torch.no_grad():  # fingerprint of the final weights (bit-identity checks between trainers)
                        metrics["param_checksum"] = float(sum(p.double().sum().item() for p in framework.model.parameters()))
                        metrics["param_abs_checksum"] = float(sum(p.double().abs().sum().item() for p in framework.model.parameters()))
                    if tr is not None and hasattr(tr, "extra_results"):
                        metrics.update(tr.extra_results())
                    for cb in (tr.callback_handler.callbacks if tr is not None else []):
                        if isinstance(cb, AccEvalCallback):
                            metrics["acc_history"] = cb.history
                metrics["config"] = {k: getattr(args, k) for k in ["task_name", "model_name", "trainer", "learning_rate", "zo_eps",
                    "max_steps", "per_device_train_batch_size", "seed", "train_set_seed", "num_train", "num_dev", "num_eval",
                    "load_float16", "load_bfloat16", "weight_decay", "zo_optimizer", "rank_r", "step_interval", "multiple_sample",
                    "num_samples", "hessian_smooth_type", "kfac_placement", "kfac_power", "kfac_damping", "kfac_block",
                    "kfac_calib_steps", "kfac_fisher", "kfac_refresh", "kfac_scope", "loren_num_queries", "loren_damping", "loren_lr_cov",
                    "loren_weight_decay", "slice_logits", "tag"]}
                _here = os.path.dirname(os.path.abspath(__file__))
                metrics["git_commit"] = os.popen(f"git -C {_here} rev-parse HEAD 2>/dev/null").read().strip()
                metrics["git_dirty"] = bool(os.popen(f"git -C {_here} status --porcelain -- '*.py' 'scripts/*' 2>/dev/null").read().strip())
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    rf = "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file
                    if os.path.dirname(rf):
                        os.makedirs(os.path.dirname(rf), exist_ok=True)
                    write_metrics_to_file(metrics, rf)
                log_final_to_wandb(metrics, args)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)

    # --- End Timer & Print Total Time ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Calculate hours, minutes, seconds for better readability
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    logger.info("=" * 30)
    logger.info(f"Total Process Finished.")
    logger.info(f"Total Wall Clock Time: {hours}h {minutes}m {seconds:.2f}s ({total_duration:.2f} seconds total)")
    logger.info("=" * 30)

if __name__ == "__main__": 
    main()