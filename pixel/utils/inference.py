from ..models.pixel.modeling_pixel import (
    PIXELForPreTraining,
    PIXELConfig,
    PIXELForSequenceClassification,
    PIXELForTokenClassification,
)
import numpy as np
import torch
from PIL import Image
from pixel_datasets.utils.dataset_utils import CustomFont
from scipy import stats as st
from skimage import data, color, transform
import cv2
from typing import Union, List
from tqdm.auto import tqdm


def preprocess_image(img):
    if type(img) == Image.Image:
        img = np.array(img)

    if type(img) == np.ndarray:
        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        if len(img.shape) == 4:
            img = img[0]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)

    return img


def load_model_for_pretraining(args, model_name):
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": args.model_revision,
        "use_auth_token": args.use_auth_token,
    }
    config = PIXELConfig.from_pretrained(
        model_name,
        attention_probs_dropout_prob=args.dropout_prob,
        hidden_dropout_prob=args.dropout_prob,
        **config_kwargs,
    )

    # Adapt config
    config.update(
        {
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": args.norm_pix_loss,
            "architectures": [PIXELForPreTraining.__name__],
        }
    )
    model = PIXELForPreTraining.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **config_kwargs,
    )
    return model


def load_model_for_squad(args, model_name):
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": args.model_revision,
        "use_auth_token": args.use_auth_token,
    }
    config = PIXELConfig.from_pretrained(
        model_name,
        attention_probs_dropout_prob=args.dropout_prob,
        hidden_dropout_prob=args.dropout_prob,
        **config_kwargs,
    )

    # Adapt config
    config.update(
        {
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": args.norm_pix_loss,
            "architectures": [PIXELForTokenClassification.__name__],
        }
    )
    model = PIXELForTokenClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **config_kwargs,
    )
    return model


def load_general_model(args, model_name, model_type: PIXELForPreTraining):
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": args.model_revision,
        "use_auth_token": args.use_auth_token,
    }
    config = PIXELConfig.from_pretrained(
        model_name,
        attention_probs_dropout_prob=args.dropout_prob,
        hidden_dropout_prob=args.dropout_prob,
        **config_kwargs,
    )

    # Adapt config
    config.update(
        {
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": args.norm_pix_loss,
            "architectures": [model_type.__name__],
        }
    )
    model = model_type.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **config_kwargs,
    )
    return model


def predict(model, image, mask):
    if type(image) == Image.Image:
        image = np.array(image)

    if type(image) == np.ndarray:
        image = np.expand_dims(image, axis=0).astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.from_numpy(image)

    elif type(image) == torch.Tensor:
        if image.max() > 1:
            image = image / 255.0
        image = image.unsqueeze(0)

    if type(mask) == np.ndarray:
        mask = mask.flatten()
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)

    elif type(mask) == torch.Tensor:
        mask = mask.flatten()
        mask = mask.unsqueeze(0)

    attention_mask = get_attention_mask(529)
    attention_mask = attention_mask.unsqueeze(0)

    model.eval()
    outputs = model(pixel_values=image, attention_mask=attention_mask, patch_mask=mask)
    return outputs


def predict_squad(model, image):
    if type(image) == Image.Image:
        image = np.array(image)

    if type(image) == np.ndarray:
        image = np.expand_dims(image, axis=0).astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.from_numpy(image)

    elif type(image) == torch.Tensor:
        if image.max() > 1:
            image = image / 255.0
        image = image.unsqueeze(0)

    attention_mask = get_attention_mask(529)
    attention_mask = attention_mask.unsqueeze(0)

    model.eval()
    outputs = model(pixel_values=image, attention_mask=attention_mask)
    return outputs


def parse_squad_outputs(outputs, original_image, labels, method):
    mask = outputs["logits"][0].detach().cpu().numpy()
    mask = np.argmax(mask, axis=1)
    mask = mask.reshape(23, 23)
    mask = np.kron(mask, np.ones((16, 16)))
    mask = np.stack([mask, mask, mask], axis=2)

    real_mask = labels.reshape(23, 23)
    real_mask = np.kron(real_mask, np.ones((16, 16)))
    real_mask = np.stack([real_mask, real_mask, real_mask], axis=2)

    if method == "masked_predictions_only":
        masked_prediction = original_image * (1 - mask) + original_image * mask * 0.6
        masked_prediction = np.clip(masked_prediction, 0, 255)
        return masked_prediction
    if method == "masked_real_only":
        masked_real = (
            original_image * (1 - real_mask) + original_image * real_mask * 0.6
        )
        masked_real = np.clip(masked_real, 0, 255)
        return masked_real
    if method == "masked_merged":
        masked = color.label2rgb(
            mask, original_image.astype("uint8"), bg_label=0, bg_color=None
        )
        masked = (masked * 255).astype(np.uint8)
        masked = color.label2rgb(
            real_mask, masked.astype, bg_label=0, bg_color=None, colors=[(0, 0, 1)]
        )
        masked = (masked * 255).astype(np.uint8)
        return masked
    if method == "saliency":
        mask = outputs["logits"][0].detach().cpu().numpy()
        mask = mask[:, 1] - mask[:, 0]
        mask = 1 / (1 + np.exp(-mask))
        mask = mask.reshape(23, 23)
        # mask = np.kron(mask, np.ones((16, 16)))
        mask = transform.resize(
            mask, (original_image.shape[0], original_image.shape[1])
        )
        heatmap = cv2.applyColorMap((mask * 255).astype("uint8"), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        merged = cv2.addWeighted(original_image.astype("uint8"), 0.5, heatmap, 0.5, 0)

        return merged


def calculate_num_patches(image):
    np_image = image.detach().cpu().numpy()
    if len(np_image.shape) == 4:
        np_image = np_image[0, 0]
    if len(np_image.shape) == 3:
        np_image = np_image[0]
    max_value = np_image.max()
    black_lines = np.any(np_image != max_value, axis=1)
    max_black_line = np.max(np.where(black_lines))
    max_black_line = max_black_line // 16
    num_patches = (max_black_line + 1) * 23
    return num_patches


def encode_images(
    model: PIXELForSequenceClassification,
    image: Union[
        Image.Image,
        np.ndarray,
        torch.Tensor,
        List[Union[Image.Image, np.ndarray, torch.Tensor]],
    ],
    batch_size: int = 16,
):
    """
    This function encodes an image or a list of images into a 768-dimensional vector or a matrix of vectors using the model.
    """
    # If the input is a single image, wrap it in a list if type(image) in [Image.Image, np.ndarray, torch.Tensor]:
    if type(image) not in [list, tuple]:
        image = [image]
    # Initialize an empty list to store the images
    images = []
    # Loop over each image in the list
    for img in image:
        # Append the image to the list
        img = preprocess_image(img)
        images.append(img)

    # Convert the list of images to a tensor
    images = torch.stack(images)

    num_patches = 529
    attention_mask = get_attention_mask(num_patches)
    attention_mask = attention_mask.repeat(len(images), 1)

    model.eval()
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    # Initialize an empty list to store the embeddings
    embeddings = []

    # Loop over the images in batches
    for i in range(0, len(images), batch_size):
        # Get the current batch of images and attention masks
        batch_images = images[i : i + batch_size]
        batch_attention_masks = attention_mask[i : i + batch_size]

        # Send the batch to the device
        batch_images = batch_images.to(model.device)
        batch_attention_masks = batch_attention_masks.to(model.device)

        # Get the model outputs
        outputs = model.vit(
            pixel_values=batch_images, attention_mask=batch_attention_masks
        )
        if model.add_cls_pooling_layer:
            sequence_output = outputs[1]
        else:
            # When not using CLS pooling mode, discard it
            sequence_output = outputs[0][:, 1:, :]
        logits = torch.mean(sequence_output, dim=1)

        # Get the embeddings and append them to the list
        batch_embeddings = logits.detach().cpu().numpy()
        embeddings.append(batch_embeddings)

    # Concatenate the list of embeddings to a numpy array
    embeddings = np.concatenate(embeddings)

    return embeddings


def parse_outputs(outputs, model, original_img):
    mask = outputs["mask"].detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping

    if type(original_img) == np.ndarray:
        original_img = torch.from_numpy(original_img.transpose(2, 0, 1))

    if original_img.max() <= 1:
        original_img = original_img * 255

    original_img_most_common = int(st.mode(original_img.numpy().flatten()).mode)

    attention_mask = outputs["attention_mask"].unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()
    predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze()
    predictions = (
        predictions * (torch.bitwise_and(mask == 1, attention_mask == 1)).long()
    )
    predictions = torch.sigmoid(predictions)
    predictions = predictions.clamp(-1, 0.5)
    predictions = predictions[0]
    predictions = torch.stack([predictions, predictions, predictions], dim=0)
    predictions = predictions - predictions.min()
    predictions = predictions / predictions.max()
    predictions = predictions * original_img_most_common
    # predictions = predictions - 0
    predictions = predictions.clamp(0, original_img_most_common)

    reconstruction = (
        original_img * (1 - (torch.bitwise_and(mask == 1, attention_mask == 1)).long())
        + predictions * mask * attention_mask
    )
    reconstruction = reconstruction.numpy()
    reconstruction = np.transpose(reconstruction, (1, 2, 0))

    reconstruction = reconstruction.astype(np.uint8)
    return reconstruction


def get_attention_mask(num_text_patches: int):
    """
    Creates an attention mask of size [1, seq_length]
    The mask is 1 where there is text or a [SEP] black patch and 0 everywhere else
    """
    n = min(num_text_patches + 1, 529)  # Add 1 for [SEP] token (black patch)
    zeros = torch.zeros(529)
    ones = torch.ones(n)
    zeros[:n] = ones
    return zeros


def get_inference_font():
    inference_font = CustomFont(
        file_name="fonts/PackardAntique-Bold.ttf",
        font_name="Packardantique-Bold",
        font_size=16,
    )
    return inference_font
