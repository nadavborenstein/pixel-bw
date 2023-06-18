import logging
from collections import defaultdict
import numpy as np

import torch
import wandb
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainingArguments,
    TrainerControl,
)

# from dataset_synthesis.synthetic_dataset import (
#     SyntheticDatasetTorch,
#     SyntheticDatasetTransform,
# )

logger = logging.getLogger(__name__)


class VisualizationCallback(TrainerCallback):
    def __init__(self, args=None, visualize_train=True, only_input=False):
        self.args = args
        self.visualize_train = visualize_train
        self.only_input = only_input

    def _clip(self, img: torch.Tensor):
        img = torch.einsum("chw->hwc", img)
        img = torch.clip(img * 255, 0, 255)
        img = torch.einsum("hwc->chw", img)
        return img

    def _log_image(self, figures_to_log: dict):
        for k, v in figures_to_log.items():
            if k in ["reconstruction", "predictions", "masked_predictions"]:
                wandb.log({k: [wandb.Image(x) for x in v]})
            else:
                wandb.log({k: [wandb.Image(self._clip(x)) for x in v]})

    def _visualize_inputs(self, model, batch):
        figures_to_log = defaultdict(list)
        for i in range(batch["pixel_values"].shape[0]):
            original_image = (
                model.unpatchify(model.patchify(batch["pixel_values"][i].unsqueeze(0)))
                .detach()
                .cpu()
                .squeeze()
            )
            figures_to_log["original_image"].append(original_image)
        self._log_image(figures_to_log)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        train_dataloader=None,
        eval_dataloader=None,
        model=None,
        **kwargs,
    ):
        logger.info(
            f"logging images. Global rank: {state.is_world_process_zero}, local rank: {state.is_local_process_zero}"
        )
        if not state.is_world_process_zero:
            return

        if self.visualize_train:
            batch = next(iter(train_dataloader))
        else:
            eval_dataloader.dataset.steps_taken = 0
            batch = next(iter(eval_dataloader))
        batch = {k: v.to(args.device) for k, v in batch.items()}
        logger.info(f"visualizing {min(32, batch['attention_mask'].shape[0])} images")

        if self.only_input:
            self._visualize_inputs(model, batch)
            return

        figures_to_log = defaultdict(list)
        model.eval()
        with torch.inference_mode():
            outputs = model(
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                patch_mask=batch["patch_mask"],
            )
        for i in range(min(len(outputs["logits"]), 32)):  # TODO don't duplicate code
            predictions = (
                model.unpatchify(outputs["logits"][i].unsqueeze(0))
                .detach()
                .cpu()
                .squeeze()
            )
            mask = outputs["mask"][i].unsqueeze(0).detach().cpu()
            mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
            mask = model.unpatchify(mask).squeeze()
            figures_to_log["mask"].append(mask)

            attention_mask = (
                batch["attention_mask"][i]
                .view(1, -1, 1)
                .detach()
                .cpu()
                .repeat(1, 1, 16**2 * 3)
            )
            attention_mask = model.unpatchify(attention_mask).squeeze()
            figures_to_log["attention_mask"].append(attention_mask)

            original_image = (
                model.unpatchify(model.patchify(batch["pixel_values"][i].unsqueeze(0)))
                .detach()
                .cpu()
                .squeeze()
            )
            figures_to_log["original_image"].append(original_image)

            im_masked = original_image * (1 - mask)
            figures_to_log["im_masked"].append(im_masked)

            figures_to_log["predictions"].append(predictions)

            masked_predictions = predictions * mask * attention_mask
            figures_to_log["masked_predictions"].append(masked_predictions)

            reconstruction = (
                original_image
                * (1 - (torch.bitwise_and(mask == 1, attention_mask == 1)).long())
                + predictions * mask * attention_mask
            )
            figures_to_log["reconstruction"].append(reconstruction)

        self._log_image(figures_to_log)
        model.train()


class SquadVisualizationCallback(TrainerCallback):
    def __init__(self, args=None, visualize_train=True, only_input=False):
        self.args = args
        self.visualize_train = visualize_train
        self.only_input = only_input

    def _log_image(self, figures_to_log: dict):
        for k, v in figures_to_log.items():
            wandb.log({k: [wandb.Image(x) for x in v]})

    def _visualize_inputs(self, model, batch):
        figures_to_log = defaultdict(list)
        for i in range(batch["pixel_values"].shape[0]):
            original_image = (
                model.unpatchify(model.patchify(batch["pixel_values"][i].unsqueeze(0)))
                .detach()
                .cpu()
                .squeeze()
            )
            figures_to_log["original_image"].append(original_image)
        self._log_image(figures_to_log)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        train_dataloader=None,
        eval_dataloader=None,
        model=None,
        **kwargs,
    ):
        logger.info(
            f"logging images. Global rank: {state.is_world_process_zero}, local rank: {state.is_local_process_zero}"
        )
        if not state.is_world_process_zero:
            return

        if self.visualize_train:
            batch = next(iter(train_dataloader))
        else:
            eval_dataloader.dataset.steps_taken = 0
            batch = next(iter(eval_dataloader))
        batch = {k: v.to(args.device) for k, v in batch.items()}
        logger.info(f"visualizing {min(32, batch['attention_mask'].shape[0])} images")

        if self.only_input:
            self._visualize_inputs(model, batch)
            return

        figures_to_log = defaultdict(list)
        model.eval()
        with torch.inference_mode():
            outputs = model(
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
            )
        for i in range(min(len(outputs["logits"]), 32)):  # TODO don't duplicate code
            mask = outputs["logits"][i].detach().cpu().numpy()
            mask = np.argmax(mask, axis=1)
            mask = mask.reshape(23, 23)
            mask = np.kron(mask, np.ones((16, 16)))
            mask = np.stack([mask, mask, mask], axis=2)
            
            real_mask = batch["labels"][i].detach().cpu().numpy()
            real_mask = real_mask.reshape(23, 23)
            real_mask = np.kron(real_mask, np.ones((16, 16)))
            real_mask = np.stack([real_mask, real_mask, real_mask], axis=2)
            
            figures_to_log["mask"].append(mask)

            original_image = batch["pixel_values"][i].detach().cpu().numpy()
            original_image = original_image.transpose(1, 2, 0)
            original_image = np.clip(original_image * 255, 0, 255)
            
            masked_prediction = original_image * (1 - mask) + original_image * mask * 0.6
            masked_prediction = np.clip(masked_prediction, 0, 255)
            
            masked_real = original_image * (1 - real_mask) + original_image * real_mask * 0.6
            masked_real = np.clip(masked_real, 0, 255)
            
            figures_to_log["original_image"].append(original_image)
            figures_to_log["masked_prediction"].append(masked_prediction)
            figures_to_log["masked_real"].append(masked_real)

        self._log_image(figures_to_log)
        model.train()


# class RandomizationIntensityEpochCallback(TrainerCallback):
#     def __init__(self, max_epochs, warmup_steps, update_type="none"):
#         self.max_epochs = max_epochs
#         self.warmup_steps = warmup_steps
#         self.update_type = update_type

#     def on_epoch_begin(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         train_dataloader=None,
#         eval_dataloader=None,
#         **kwargs,
#     ):
#         current_epoch = state.epoch

#         train_dataset: SyntheticDatasetTorch = train_dataloader.dataset
#         train_args: RenderingArguments = train_dataset.args

#         train_args.update_randomness_intensity(
#             max_step=self.max_epochs,
#             step=current_epoch,
#             update_type=self.update_type,
#             warmup_steps=self.warmup_steps,
#         )

#         logger.info(f"updating randomness intensity: {current_epoch}/{self.max_epochs} with {self.warmup_steps} warmup epochs. Update type: {self.update_type}")

#         train_dataloader.dataset.transform = SyntheticDatasetTransform(train_args, rng=train_dataset.rng)
#         train_dataloader.dataset.args = train_args

#         if eval_dataloader is not None:
#             eval_dataset: SyntheticDatasetTorch = eval_dataloader.dataset
#             eval_args: RenderingArguments = eval_dataset.args
#             eval_args.update_randomness_intensity(
#                 max_step=self.max_epochs,
#                 step=current_epoch,
#                 update_type=self.update_type,
#                 warmup_steps=self.warmup_steps,
#             )
#             eval_dataloader.dataset.transform = SyntheticDatasetTransform(eval_args, rng=eval_dataset.rng)
#             eval_dataloader.dataset.args = eval_args
