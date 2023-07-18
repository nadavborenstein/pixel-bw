import transformers
from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

squad_v2 = True
model_checkpoint = "bert-base-cased"
batch_size = 16
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.


datasets = load_dataset("squad_v2" if squad_v2 else "squad")
runaway_dataset = load_from_disk("Runaway_dataset")

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def balance_dataset(dataset, oversample=False):
    dataset_1 = dataset.filter(lambda example: len(example["answers"]["text"]) > 0)
    dataset_0 = dataset.filter(lambda example: len(example["answers"]["text"]) == 0)
    min_len = min(len(dataset_1), len(dataset_0))
    max_len = max(len(dataset_1), len(dataset_0))
    
    if oversample:
      if dataset_1.num_rows < max_len:
        num_repeats = max_len // dataset_1.num_rows
        dataset_1 = concatenate_datasets([dataset_1] * num_repeats)
      elif dataset_0.num_rows < max_len:
        num_repeats = max_len // dataset_0.num_rows
        dataset_0 = concatenate_datasets([dataset_0] * num_repeats)
    else:
      dataset_1 = dataset_1.shuffle(seed=42).select(range(min_len))
      dataset_0 = dataset_0.shuffle(seed=42).select(range(min_len))
    return concatenate_datasets([dataset_1, dataset_0])



runaway_dataset["test"] = balance_dataset(runaway_dataset["test"])
runaway_dataset["train"] = balance_dataset(runaway_dataset["train"], oversample=True)
print(runaway_dataset)

    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
pad_on_right = tokenizer.padding_side == "right"
tokenized_datasets = runaway_dataset.map(prepare_train_features, batched=True, remove_columns=runaway_dataset["train"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-runaways",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

data_collator = default_data_collator
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("test-runaways-trained")
