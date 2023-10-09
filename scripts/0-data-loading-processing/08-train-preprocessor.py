"""Training preprocessor for gene expression prediction.
"""
import pickle
from functools import reduce, partial
import operator

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

from module.florabert import config, utils


DATA_DIR = config.data_final / "nam_data"
TRAIN_DATA = "merged_seq_genex.csv"
SAVE_PATH = config.models / "preprocessor" / "preprocessor.pkl"


def load_data(args):
    data = pd.read_csv(args.train_data, nrows=10000)
    tissues = args.tissue_subset or config.tissues
    data = data.loc[data.fold != "test", tissues]
    if args.threshold:
        cond = reduce(
            lambda x, y: x | y, [data[tiss] >= args.threshold for tiss in tissues]
        )
        data = data[cond]
    return data

def apply_log1p_with_offset(x, offset):
    # Apply log1p with an offset to handle non-positive values
    if offset is not None:
        return np.log1p(np.maximum(x, 0.0)) + offset
    else:
        return x

def load_preprocessor(args):
    steps = []
    if args.log_offset:
        # If you want to apply log and add an offset, you can do it for the entire DataFrame
        steps.append(FunctionTransformer(func=partial(np.add, args.log_offset)))
        
    log_offset = args.log_offset if hasattr(args, 'log_offset') else 0.0

    # You may or may not need StandardScaler, depending on your specific use case
    steps.append(FunctionTransformer(func=apply_log1p_with_offset, kw_args={"offset": log_offset}, validate=False))
    # if args.scale_data:
    steps.append(StandardScaler(with_mean=True, with_std=True))

    return make_pipeline(*steps)



def fit_preprocessor(data, preprocessor):
    print(data)
    return preprocessor.fit(data)


def main():
    args = utils.get_args(data_dir=DATA_DIR, train_data=TRAIN_DATA)
    data = load_data(args)
    preprocessor = load_preprocessor(args)
    preprocessor = fit_preprocessor(data, preprocessor)
    with SAVE_PATH.open("wb") as f:
        pickle.dump(preprocessor, f)


if __name__ == "__main__":
    main()
