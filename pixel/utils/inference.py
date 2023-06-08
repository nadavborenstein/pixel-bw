from ..models.pixel.modeling_pixel import PIXELForPreTraining, PIXELConfig
import numpy as np
import torch


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


def predict(model, image, mask):
    if type(image) == np.ndarray:
        image = np.expand_dims(image, axis=0).astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.from_numpy(image)

    elif type(image) == torch.Tensor:
        image = image.unsqueeze(0)

    if type(mask) == np.ndarray:
        mask = mask.flatten()
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)

    elif type(mask) == torch.Tensor:
        mask = mask.flatten()
        mask = mask.unsqueeze(0)

    attention_mask = get_attention_mask(529)
    attention_mask = torch.unsqueeze(attention_mask, 0)

    model.eval()
    outputs = model(pixel_values=image, attention_mask=attention_mask, patch_mask=mask)
    return outputs


def parse_outputs(outputs, model, original_img):
    mask = outputs["mask"].detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping

    attention_mask = outputs["attention_mask"].unsqueeze(-1).repeat(1, 1, 16**2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()
    predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze()
    predictions = (
        predictions * (torch.bitwise_and(mask == 1, attention_mask == 1)).long()
    )
    predictions = torch.sigmoid(predictions)
    predictions = predictions.clamp(-1, 0.5)
    predictions = predictions - predictions.min()
    predictions = predictions / predictions.max()
    predictions = predictions * 255
    predictions = predictions - 0
    predictions = predictions.clamp(0, 255)

    if type(original_img) == np.ndarray:
        original_img = torch.from_numpy(original_img.transpose(2, 0, 1))

    if original_img.max() <= 1:
        original_img = original_img * 255

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
