import argparse
from kasafranse.preprocessing import Preprocessing
from kasafranse.hugging_face_utils import BuildDataset
from transformers import AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq,\
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from transformers import AutoTokenizer
import torch
import numpy as np

from datasets import load_metric

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Fine-turned Pretrained Huging Face model with Custom Dataset')
    parser.add_argument("pretrained_model",
                        help="Provide the Pretrained model to be Fineturned", type=str)
    parser.add_argument("input_train_data",
                        help="Provide the Path to the Input language Training  Data", type=str)
    parser.add_argument("target_train_data",
                        help="Provide the Path to the Target Language Training  Data", type=str)
    parser.add_argument(
        "input_val_data", help="Provide the Path to the Input language validation Data", type=str)
    parser.add_argument(
        "target_val_data", help="Provide the Path to the Target Language validation Data", type=str)
    parser.add_argument(
        "src_lang", help="Provide the Initial of the Source Language as used in the Pretrained model.\
            Example the source language of the pretrained model 'Helsinki-NLP/opus-mt-en-tw' is en", type=str)
    parser.add_argument(
        "targ_lang", help="Provide the Initial of the Target Language as used in the Pretrained model.\
            Example the Target language of the pretrained model 'Helsinki-NLP/opus-mt-en-tw' is tw", type=str)
    parser.add_argument(
        "--input_max_length", type=int, default=50, help="Enter the maximum length for the source language")
    parser.add_argument(
        "--output_max_lenght", type=int, default=50, help="Enter the maximum length for the target language")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Give the batch size for traininf")
    parser.add_argument(
        "--epoch", type=int, default=20, help="Enter the number of epoch for training")
    parser.add_argument('--savedir', type=str,
                        help="Provide the path to save to the fineturned model")
    parser.add_argument("model_name", type=str,
                        help="Name for saving the fineturend model")

    args = parser.parse_args()
    print("MODEL PATH: " + args.savedir)
    print("FINED-TURNED MODEL NAME: " + args.model_name)
    print("================")

    pretrained_model = args.pretrained_model
    src_lang_train_path = args.input_train_data
    targ_lang_train_path = args.target_train_data
    src_lang_val_path = args.input_val_data
    targ_lang_val_path = args.target_val_data
    src = args.src_lang
    targ = args.targ_lang
    max_input_length = args.input_max_lenght
    max_target_length = args.output_max_lenght
    batch_size = args.batch_size
    epoch = args.epoch

    # Provide the pretrained model
    model_checkpoint = pretrained_model

    # load the metrics
    metric = load_metric("sacrebleu")

    # Load the datasets
    # Create an instance of tft preprocessing class
    preprocessor = Preprocessing()

    # Read training parallel dataset
    src_train_data, targ_train_data = preprocessor.read_parallel_dataset(
        filepath_1=src_lang_train_path,
        filepath_2=targ_lang_train_path
    )

    # Read validation parallel dataset
    src_val_data, targ_val_data = preprocessor.read_parallel_dataset(
        filepath_1=src_lang_val_path,
        filepath_2=targ_lang_val_path
    )

    # Build Customize Hugging Face Dataset
    dataset = BuildDataset(
        src_train_data, targ_train_data, src_val_data, targ_val_data, src, targ)

    train_dataset, val_dataset = dataset.build()
    # load tokenizer of pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Define function to preprocess the input and target data
    max_input_length = max_input_length
    max_target_length = max_target_length
    source_lang = src
    target_lang = targ

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # preprocess datasets
    tokenized_train_datasets = train_dataset.map(
        preprocess_function, batched=True)
    tokenized_val_datasets = val_dataset.map(preprocess_function, batched=True)

    # Download model weights
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # provide training configs
    batch_size = batch_size
    model_name = model_checkpoint.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epoch,
        predict_with_generate=True
    )

    # set data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instaintiate transformer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_val_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Begin Training
    trainer.train()

    # save fineturned model
    if args.savedir:
        trainer.save_model(f"{args.savedir}/{model_name}")
    else:
        trainer.save_model("{model_name}")
