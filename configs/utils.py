from wandb.sdk.wandb_config import Config
import torch
import torch.distributed as dist
import sys
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

RETRACTED_MARKER = "<RETRACTED>"
OPTIONAL_MARKER = "<OPTIONAL>"
NON_OPTIONAL_MARKER = "<NON_OPTIONAL>"

RETRACTED_VALUES = ["use_auth_token", "hub_token"]
NON_OPTIONAL_VALUES = ["do_train"]
NON_OPTIONAL_VALUES = ["run_name"]


def update_config_key(config: Config, key: str, value: str):
    """
    This function updates the config with the update_dict.
    """
    config.update({key: value}, allow_val_change=True)
    return config


def generate_training_args_from_config(config: Config):
    training_args = TrainingArguments(output_dir=config.output_dir)
    training_args.__dict__.update(config)
    return training_args


def read_args():
    # create an empty dictionary
    args = {}
    # loop through the command line arguments starting from the second one
    for arg in sys.argv[1:]:
        # split the argument by '=' sign
        key, value = arg.split("=")
        # remove any leading or trailing spaces
        key = key.strip()
        value = value.strip()
        # remove any leading '-' signs from the key
        key = key.lstrip("-")
        # add the key-value pair to the dictionary
        args[key] = value
    # return the dictionary
    return args


def evaluate_config_before_update(config: Config):
    """
    This function checks if the config contains any retracted values.
    """
    for key in config.keys():
        if key in RETRACTED_VALUES:
            assert (
                config[key] == RETRACTED_MARKER
            ), "potential security issue: config[{}] = {}".format(key, config[key])
        if key in NON_OPTIONAL_VALUES and key in config.keys():
            assert config[key] != OPTIONAL_MARKER, "potential utiliy issue: {}".format(
                key
            )
    return True


def evaluate_config_after_update(config: Config):
    for key in config.keys():
        if key in RETRACTED_VALUES:
            assert config[key] != RETRACTED_MARKER, "Please update config[{}]".format(
                key
            )
        if key in NON_OPTIONAL_VALUES:
            assert (
                config[key] != NON_OPTIONAL_MARKER
            ), "Please update config[{}], it is a non-optional value".format(key)
    return True


def update_config(config: Config, update_dict: dict):
    """
    This function updates the config with the update_dict.
    """

    config.update(update_dict, allow_val_change=True)

    training_args = TrainingArguments(output_dir=config.output_dir)
    training_args = training_args.to_dict()
    for key in training_args.keys():
        if key not in config.keys():
            update_config_key(config, key, training_args[key])

    update_config_key(config, "push_to_hub_token", config["hub_token"])
    update_config_key(config, "n_gpu", config["_n_gpu"])

    for key in config.keys():
        if key in NON_OPTIONAL_VALUES and config[key] == OPTIONAL_MARKER:
            update_config_key(config, key, False)
        if config[key] == "none" or config[key] == "None":
            update_config_key(config, key, None)
        if config[key] == "true" or config[key] == "True":
            update_config_key(config, key, True)
        if config[key] == "false" or config[key] == "False":
            update_config_key(config, key, True)
    return config
