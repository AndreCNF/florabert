from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import *

from module import config, transformers_utility as tr, utils, metrics, dataio
# from prettytable import PrettyTable
import numpy as np

app = FastAPI()
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.mount("FloraBERT.static", StaticFiles(directory="FloraBERT.static"), name="static")
templates = Jinja2Templates(directory="templates")

# table = PrettyTable()
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PRETRAINED_MODEL = config.models / "transformer" / "prediction-model" / "saved_model.pth"
DATA_DIR = config.data

def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return {"filename": file.filename}

@app.get("/process/{filename}", response_class=HTMLResponse)
def process_file(request: Request, filename: str):
    file_path = DATA_DIR / filename
    preds = main(
        data_dir=DATA_DIR,
        train_data=file_path,
        test_data=file_path,
        pretrained_model=PRETRAINED_MODEL,
        tokenizer_dir=TOKENIZER_DIR,
        model_name="roberta-pred-mean-pool",
    )
    predictions = []
    for i in range(len(preds)):
        predictions.append([{"tissue": config.tissues[j], "prediction": preds[i][j] } for j in range(8)])
    # print(predictions)
    return JSONResponse(content=predictions)

def main(data_dir: str, train_data: str, test_data: str, pretrained_model: str, tokenizer_dir: str, model_name: str):
    args = utils.get_args(
        data_dir=data_dir,
        train_data=train_data,
        test_data=test_data,
        pretrained_model=pretrained_model,
        tokenizer_dir=tokenizer_dir,
        model_name=model_name,
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
    # print(preds)
    # for e in preds:
    #     table.add_row(e)
    # print(table)

    return preds.tolist()
