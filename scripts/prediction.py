from module.florabert import config, transformers as tr, utils, metrics, dataio
from prettytable import PrettyTable
import numpy as np

table = PrettyTable()
table.field_names = config.tissues
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PRETRAINED_MODEL = (
    config.models / "transformer" / "prediction-model" / "pytorch_model.bin"
)
DATA_DIR = config.data


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )


def main(TEST_DATA):
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=TEST_DATA,
        test_data=TEST_DATA,
        pretrained_model=PRETRAINED_MODEL,
        tokenizer_dir=TOKENIZER_DIR,
        model_name="roberta-pred-mean-pool",
    )

    settings = utils.get_model_settings(config.settings, args)
    if args.output_mode:
        settings["output_mode"] = args.output_mode
    if args.tissue_subset is not None:
        settings["num_labels"] = len(args.tissue_subset)

    print("Loading model...")
    config_obj, tokenizer, model = load_model(args, settings)

    print("Loading data...")
    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        eval_data=args.eval_data,
        test_data=args.test_data,
        seq_key="text",
        file_type="text",
        filter_empty=args.filter_empty,
        shuffle=False,
    )
    dataset_test = datasets["train"]

    print("Getting predictions:")
    preds = np.exp(np.array(metrics.get_predictions(model, dataset_test))) - 1
    for e in preds:
        table.add_row(e)
    print(table)


if __name__ == "__main__":
    main("test.txt")
