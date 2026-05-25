#!/usr/bin/env python3

"""
Compute SSL features for LibriSpeech.
Manifests are read from data/manifests.
Features are saved under data/{model_tag}_{layer}.
"""

import argparse
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
from dataclasses import dataclass
from pathlib import Path
import sentencepiece as spm

import numpy as np
import torch
from filter_cuts import filter_cuts

from lhotse import CutSet, LilcomChunkyWriter
from lhotse.features.base import FeatureExtractor
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import compute_num_frames

from icefall.utils import Optional, get_executor, str2bool
from transformers import Wav2Vec2FeatureExtractor, HubertModel

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_REGISTRY = {
    
    "hubert-base": {
        "hf_name": "facebook/hubert-base-ls960",
        "hidden_size": 768,
        "num_layers": 12,
        "tag": "hubert-base",
    },
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="SSL model to use: currently only supports 'hubert-base'",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Transformer layer to extract (0-indexed). "
             "hubert-base: 0-11",
    )
 
    return parser.parse_args()


_model = None
_feature_extractor = None
_loaded_model_name = None

def _get_model_and_extractor(hf_name: str, layer: int):
    global _model, _feature_extractor, _loaded_model_name
    if _model is None or _loaded_model_name != hf_name:
        logging.info(f"Loading {hf_name} on {device} ...")
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_name)
        _model = HubertModel.from_pretrained(hf_name)
        _model.to(device)
        _model.eval()
        _model.encoder.layers = _model.encoder.layers[:layer]
        _loaded_model_name = hf_name
        logging.info("Model loaded.")
    return _model, _feature_extractor


@dataclass
class SSLConfig:
    hf_name: str = "facebook/hubert-base-ls960"
    layer: int = 9
    hidden_size: int = 768
    sampling_rate: int = 16_000
    model_tag: str = "hubert-base"


class SSLExtractor(FeatureExtractor):
    name = SSLConfig.model_tag
    config_type = SSLConfig

    def __init__(self, config: Optional[SSLConfig] = None):
        super().__init__(config if config is not None else SSLConfig())

    @property
    def frame_shift(self) -> float:
        return 0.02  # 20ms

    @property
    def feature_dim(self) -> int:
        return self.config.hidden_size

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        assert sampling_rate == self.config.sampling_rate, (
            f"Expected {self.config.sampling_rate} Hz, got {sampling_rate} Hz"
        )

        model, feature_extractor = _get_model_and_extractor(self.config.hf_name, self.config.layer)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples).float()
        if samples.dim() == 2:
            samples = samples.squeeze(0)

        num_samples = samples.shape[0]

        inputs = feature_extractor(
            samples.numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)

        feats = outputs.hidden_states[self.config.layer].squeeze(0).cpu().numpy()

        expected_frames = compute_num_frames(
            duration=num_samples / sampling_rate,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
        )
        if feats.shape[0] < expected_frames:
            pad = expected_frames - feats.shape[0]
            feats = np.pad(feats, ((0, pad), (0, 0)), mode="edge")
        else:
            feats = feats[:expected_frames]

        return feats


def compute_ssl_librispeech(
    model_key: str,
    layer: int,
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    registry = MODEL_REGISTRY[model_key]
    hf_name = registry["hf_name"]
    hidden_size = registry["hidden_size"]
    num_layers = registry["num_layers"]
    model_tag = registry["tag"]

    assert 0 <= layer < num_layers, (
        f"Layer {layer} out of range for {hf_name} (0-{num_layers - 1})"
    )

    src_dir = Path("data/manifests")
    output_dir = Path(f"data/{model_tag}_{layer}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Model: {hf_name}, layer: {layer}, output: {output_dir}")

    config = SSLConfig(
        hf_name=hf_name,
        layer=layer,
        hidden_size=hidden_size,
        model_tag=model_tag,
    )
    
    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
       
    extractor = SSLExtractor(config=config)
    
    if dataset is None:
        dataset_parts = (
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "librispeech"
    suffix = "jsonl.gz"

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    
    assert manifests is not None

    num_jobs = 1

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
                if perturb_speed:
                    logging.info(f"Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(vars(args))

    compute_ssl_librispeech(
        model_key=args.model,
        layer=args.layer,
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
    )
