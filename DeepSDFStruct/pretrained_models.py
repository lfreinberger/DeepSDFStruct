from enum import Enum
import importlib.resources
from DeepSDFStruct.deep_sdf.workspace import load_trained_model, load_latent_vectors
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
import torch


class PretrainedModels(Enum):
    ChiAndCross = "chi_and_cross"
    AnalyticRoundCross = "analytic_round_cross"
    RoundCross = "round_cross"
    # The primitive decoders differ only in latent code length. ``Primitives``
    # is the default and aliases the widest (and best performing) one, so
    # ``PrimitivesCL32 is PretrainedModels.Primitives``.
    Primitives = "primitives_cl32"
    PrimitivesCL32 = "primitives_cl32"
    PrimitivesCL16 = "primitives_cl16"
    PrimitivesCL08 = "primitives_cl08"
    Primitives2D = "primitives_2d"


# Maps enum entries to file paths
main_dir = importlib.resources.files("DeepSDFStruct")
_MODEL_REGISTRY = {
    model: main_dir / "trained_models" / model.value for model in PretrainedModels
}


def get_model(
    model: str | PretrainedModels, checkpoint: str = "latest", device=None
) -> DeepSDFModel:
    """
    Load a pretrained model by name or enum.

    Args:
        model (str | PretrainedModels): model identifier
        checkpoint (str): checkpoint file name (default: 'latest')

    Returns:
        Trained PyTorch model
    """
    if isinstance(model, str):
        path = model
    else:
        model_enum = model
        path = _MODEL_REGISTRY.get(model_enum)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not path:
        raise ValueError(f"Model path {path} not found.")
    decoder = load_trained_model(path, checkpoint, device=device)
    latent_vectors = load_latent_vectors(path, checkpoint, device=device)
    decoder.eval()
    deep_sdf_model = DeepSDFModel(decoder, latent_vectors, device=device)
    return deep_sdf_model


def list_available_models():
    return list(PretrainedModels)
