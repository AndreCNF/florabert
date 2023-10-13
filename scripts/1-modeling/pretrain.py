# """
# Pretraining on masked language model task.
# """
# import torch

# from module.florabert import config, utils, training, dataio
# from module.florabert import transformers as tr

# DATA_DIR = config.data_final / "transformer" / "seq"
# TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"

# OUTPUT_DIR = config.models / "transformer" / "language-model"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# def main():
#     args = utils.get_args(
#         data_dir=DATA_DIR,
#         train_data="all_seqs_train.txt",
#         test_data="all_seqs_test.txt",
#         output_dir=OUTPUT_DIR,
#         model_name="roberta-lm",
#     )
#     print(args)

#     settings = utils.get_model_settings(config.settings, args, args.model_name)


#     config_obj, tokenizer, model = tr.load_model(
#         args.model_name,
#         TOKENIZER_DIR,
#         pretrained_model=args.pretrained_model,
#         **settings,
#     )

#     num_params = utils.count_model_parameters(model, trainable_only=True)
#     print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

#     datasets = dataio.load_datasets(
#         tokenizer,
#         args.train_data,
#         test_data=args.test_data,
#         file_type="text",
#         seq_key="text",
#     )
#     dataset_train = datasets["train"]
#     dataset_test = datasets["test"]
#     print(f"Loaded training data with {len(dataset_train):,} examples")
#     data_collator = dataio.load_data_collator(
#         "language-model",
#         tokenizer=tokenizer,
#     )

#     training_settings = config.settings["training"]["pretrain"]
#     trainer = training.make_trainer(
#         model,
#         data_collator,
#         dataset_train,
#         dataset_test,
#         args.output_dir,
#         **training_settings,
#     )

#     print(f"Starting training on {torch.cuda.device_count()} GPUs")
#     training.do_training(trainer, args, args.output_dir)

#     print("Saving model")

#     trainer.save_model(str(args.output_dir))


# if __name__ == "__main__":
#     main()
import torch
from module.florabert import config, utils, training, dataio
from module.florabert import transformers as tr
import os

input_number = int(os.environ.get("INPUT_NUMBER", 0))

DATA_DIR = config.data_final / "transformer" / "seq"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
OUTPUT_DIR = config.models / "transformer" / "language-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=f"all_seqs_train_{input_number}.txt",  # Replace with the path to your first part of the dataset
        test_data="all_seqs_test.txt",  # Replace with the path to the corresponding test data
        output_dir=OUTPUT_DIR,  # Save checkpoints and logs in a separate directory
        model_name="roberta-lm",
    )
    print(args)

    settings = utils.get_model_settings(config.settings, args, args.model_name)

    config_obj, tokenizer, model = tr.load_model(
        args.model_name,
        TOKENIZER_DIR,
        pretrained_model=args.pretrained_model,
        **settings,
    )

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        test_data=args.test_data,
        file_type="text",
        seq_key="text",
    )
    dataset_train = datasets["train"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train):,} examples")
    data_collator = dataio.load_data_collator(
        "language-model",
        tokenizer=tokenizer,
    )

    training_settings = config.settings["training"]["pretrain"]
    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        **training_settings,
    )

    print(f"Starting training on {torch.cuda.device_count()} GPUs")
    training.do_training(trainer, args, args.output_dir)

    print("Saving model")
    trainer.save_model(str(args.output_dir))

if __name__ == "__main__":
    main()
