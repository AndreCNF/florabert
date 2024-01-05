"""
Fine-tuning the transformer model on the downstream gene expression prediction task
"""
import os
import sys
sys.path.append('/kaggle/working/florabert')
import torch
import multiprocessing
import numpy as np
from torch.utils.data import DataLoader
# import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr
from module.florabert.utils import get_latest_checkpoint, compute_r2, compute_mse

NUM_REPLICAS = 2

def make_search_space(trial) -> dict:
    return {
        "learning_rate": tune.loguniform(1e-5, 1e-4),
        "num_train_epochs": tune.choice(range(1, 20)),
        "seed": tune.choice(range(1, 41)),
        # "per_device_train_batch_size": 64,
        # "delay_size": tune.choice(range(0, 750)),
        # "betas": [0.99, 0.999],
        # "eps": 1e-8,
        # # "weight_decay": tune.uniform(0, 1),
        # "weight_decay": 0,
        # "warmup_steps": tune.choice(range(0, 200)),
        # "num_param_groups": 2,
    }


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )

DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TRAIN_DATA = "train.tsv"
EVAL_DATA = "eval.tsv"
TEST_DATA = "test.tsv"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PREPROCESSOR = config.models / "preprocessor" / "preprocessor.pkl"

# Starting from last checkpoint of the general purpose model
PRETRAINED_MODEL = (
    config.models / "transformer" / "language-model"
)
OUTPUT_DIR = config.models / "transformer" / "prediction-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _mp_fn():
    accelerator = Accelerator(mixed_precision = 'fp16')
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=TRAIN_DATA,
        eval_data=EVAL_DATA,
        test_data=TEST_DATA,
        output_dir=OUTPUT_DIR,
        pretrained_model=PRETRAINED_MODEL,
        tokenizer_dir=TOKENIZER_DIR,
        model_name="roberta-pred-mean-pool",
        log_offset=1,
        preprocessor=PREPROCESSOR,
        transformation="log",
        hyperparam_search_metrics="mse",
        hyperparam_search_trials=10,
    )
    settings = utils.get_model_settings(config.settings, args)

#     print(f"Model settings: {settings}")
    print("Making model")
    config_obj, tokenizer, model = load_model(args, settings)
    if args.freeze_base:
        print("Freezing base")
        utils.freeze_base(model)

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")
    print("Loading data")

    preprocessor = utils.load_pickle(args.preprocessor) if args.preprocessor else None
    with accelerator.main_process_first():
        datasets = dataio.load_datasets(
            tokenizer,
            args.train_data,
            eval_data=args.eval_data,
            test_data=args.test_data,
            seq_key="sequence",
            file_type="csv",
            delimiter="\t",
            log_offset=args.log_offset,
            preprocessor=preprocessor,
            filter_empty=args.filter_empty,
            tissue_subset=args.tissue_subset,
            threshold=args.threshold,
            transformation=args.transformation,
            discretize=(args.output_mode == "classification"),
            nshards=args.nshards,
        )

    dataset_train = datasets["train"].remove_columns(['sequence'])
    dataset_test = datasets["eval"].remove_columns(['sequence'])
    print(f"Loaded training data with {len(dataset_train)} examples")

    if args.nshards_eval:
        print(f"Keeping shard 1/{args.nshards_eval} of eval data")
        dataset_eval = dataset_eval.shard(num_shards=args.nshards_eval, index=1)

    data_collator = dataio.load_data_collator("pred")
    training_settings = config.settings["training"]["finetune"]
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    print(training_settings)
    
    train_dataloader = DataLoader(
        dataset_train, batch_size=64, collate_fn = data_collator, shuffle = True
    )
    eval_dataloader = DataLoader(
        dataset_test, batch_size=8, collate_fn = data_collator, shuffle = False
    )
    
    model_init = lambda: load_model(args, settings)[2]  # For hyperparameter search
    optimizers = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        hyperparameter_search=args.hyperparameter_search,
        model_init=model_init,
        metrics=args.metrics,
        **training_settings,
    )
    progress_bar = tqdm(range(int(3 * len(train_dataloader) / NUM_REPLICAS)))
    optimizer, scheduler = optimizers
    train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
     train_dataloader, eval_dataloader, model, optimizer, scheduler
    )
    accelerator.print("Starting training")
    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)

        model.eval()
        all_predictions = []
        all_labels = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits
            all_predictions.append(accelerator.gather(predictions))
            all_labels.append(accelerator.gather(batch["labels"]))
            
        all_predictions = torch.cat(all_predictions)[:len(datasets["eval"])]
        all_labels = torch.cat(all_labels)[:len(datasets["eval"])]

        eval_metric = compute_mse(all_labels, all_predictions)

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            OUTPUT_DIR / f"epoch_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        
    print("Saving model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        OUTPUT_DIR / "final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


if __name__ == "__main__":
    _mp_fn()
