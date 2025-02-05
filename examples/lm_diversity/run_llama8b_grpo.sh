set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/panorama/rl/train.parquet \
    data.val_files=data/panorama/rl/val.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.return_raw_input_ids=False \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.5 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.decode_responses=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=meta-llama/Llama-3.1-8B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.cliprange_value=0.1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_ctrl.type=fixed \
    reward_model.enable=True \
    reward_model.model.path=allenai/Llama-3.1-Tulu-3-8B-RM \
    reward_model.model.input_tokenizer=allenai/Llama-3.1-Tulu-3-8B-RM \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lm_diversity' \
    trainer.experiment_name='lm_diversity_llama8b_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 $@





    # reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    # reward_model.model.input_tokenizer=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
#   64 for 8B


