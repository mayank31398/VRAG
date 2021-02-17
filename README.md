#### Flags used to run the code:
1. `--params_file` Path to params file
1. `--eval_only` If passed, models only generate responses (no training is done)
1. `--checkpoint` Checkpoint name to load from
1. `--knowledge_file` Path to knowledge.jsonl
1. `--labels_file` File to evaluate (only to be used when --eval_only is used)
1. `--output_file` Path to dump generated outputs (only to be used when --eval_only is used)
1. `--model_path` Path to save/load the model from
1. `--prior_path` Path to load the prior from (only use if model_path is not specified)
1. `--posterior_path` Path to load the posterior from (only use if model_path is not specified)
1. `--decoder_path` Path to load the decoder from (only use if model_path is not specified)
1. `--build_index` Builds index and exits (needs to be run before training)
1. `--n_gpus` Number of GPUs to use (defaults to 1) (May not work)
1. `--dialog` To be passed if dataset is a dialog dataset
1. `--save_every` Save every nth step (Never tested this argument so don't use. By default saves on each epoch)
1. `--multitask` Pass to use a classifier loss with decoder to classify CANNOTANSWER instances
1. `--weight` Weight for CANNOTANSWER instances (only use if --weigh_cannot_answer is passed)
1. `--weigh_cannot_answer` Weigh CANNOTANSWER instances
1. `--skip_cannot_answer` Skips CANNOTANSWER instances
1. `--fix_DPR` Fix both prior and posterior during training
1. `--fix_prior` Fix only prior during training
1. `--fix_posterior` Fix only posterior during training
1. `--fix_decoder` Fix only decoder during training

Look at [train.sh](train.sh) and [val.sh](val.sh) for how to train and evaluate the models.