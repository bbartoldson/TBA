# Trajectory Balance with Asynchrony

This is the official repository for the paper [*Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training*](https://arxiv.org/pdf/2503.18929).

Our async RL approach speeds up various LLM post-training pipelines (GSM8K, TL;DR, red-teaming). You can use the code here to reproduce results like the following.

![Screenshot 2025-04-09 at 12 40 30 PM](https://github.com/user-attachments/assets/c19e3548-072d-4e2c-b040-4d8e2021a5c2)

## TBA′

**TBA′** is our TBA implementation for larger-scale training with 7+ billion parameter models, with evaluation on MATH 500 and reasoning gym tasks.

For the TBA′ code, visit the [TBA-prime repository](https://github.com/bbartoldson/TBA-prime).

# Setup

Install the necessary requirements.

```pip install -r requirements.txt```

To exactly reproduce our results, please use a 4xA100 (80 GB) node.

If you have different hardware, just modify the relevant arguments in one of the launch scripts. For example, with `launch_training_gsm8k.sh`,

   1. Set `num_processes` to your GPU count (summing across all nodes).
   2. In the `srun` command, change the value for the `N` parameter to your node count. E.g., if you have 2 nodes, use `-N 2`.

We assume either `srun` or `mpirun` is available on your system. The launch script autodetects which you have to choose the appropriate command.

# Training Models

To launch a GSM8K experiment, run the following command.

   ```sh launch_training_gsm8k.sh```

To launch a TL;DR experiment, run the following command.

   ```sh launch_training_tldr.sh```


# Evaluating Results

The GSM8K test performance will be generated automatically at the end of the training script.

For TL;DR, we leverage the post-training evaluation pipeline of Noukhovitch et al. (2025). To evaluate a trained model, use the following 2 steps.

1. Remove "module." from the state dict's keys.

     ```python eval_tldr/process_checkpoint.py results/tldr_model_size_1b_run_0/checkpoint-256/```
2. Run evaluations using the checkpoint from step 1.

     ```sh eval_tldr/run.sh results/tldr_model_size_1b_run_0/checkpoint-256-NM```

# Acknowledgements

This repository leverages prior work of [mnoukhov](https://github.com/mnoukhov/async_rlhf), who provides TL;DR SFTed models and evaluation code. They credit [Costa](https://github.com/vwxyzjn) for `src/vllm_utils.py`, which isolates vLLM instances to facilitate asynchrony. We also leverage the TL;DR reward model of [Costa et al.](https://github.com/vwxyzjn/summarize_from_feedback_details). Finally, we utilize the GSM8K SFTed model of [kazemnejad](https://github.com/McGill-NLP/VinePPO).


# Citation

```
@article{bartoldson2025trajectory,
  title={Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training},
  author={Bartoldson, Brian R and Venkatraman, Siddarth and Diffenderfer, James and Jain, Moksh and Ben-Nun, Tal and Lee, Seanie and Kim, Minsu and Obando-Ceron, Johan and Bengio, Yoshua and Kailkhura, Bhavya},
  journal={arXiv preprint arXiv:2503.18929},
  year={2025}
}
```

# Release

The code of this site is released under the Apache 2.0 license.

LLNL-CODE-2004475
