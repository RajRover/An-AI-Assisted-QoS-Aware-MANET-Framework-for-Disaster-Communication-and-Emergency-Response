"""
classifier.py

Production inference engine for the AI-Assisted QoS-Aware MANET Framework
for Disaster Communication.

Wraps a pre-trained Hugging Face sequence-classification model
(models/multi_class_disaster_model/) and its label map
(models/class_map.json) behind a single, reusable, thread-safe-by-design
API: `DisasterClassifier`.

This module performs INFERENCE ONLY. It does not train, fine-tune, or
modify model weights, and it does not expose any web framework (no
Streamlit, FastAPI, or Flask code lives here). It is meant to be imported
directly by downstream components of the MANET framework -- the Packet
Generator, QoS Mapper, Routing Engine, Traffic Generator, and Simulation
Runner -- so its only job is to turn a message into a disaster-category
prediction, quickly and predictably.

Author: ML Engineering & Software Architecture
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================================
# MODULE LOGGER
# ============================================================================

logger = logging.getLogger("DisasterMANET.DisasterClassifier")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ============================================================================
# PATH DEFAULTS
# ============================================================================

# classifier.py lives at Disaster_Prediction/classifier.py; the model
# artifacts live alongside it under Disaster_Prediction/models/.
_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = _MODULE_DIR / "models" / "multi_class_disaster_model"
DEFAULT_CLASS_MAP_PATH = _MODULE_DIR / "models" / "class_map.json"


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DisasterClassifierError(Exception):
    """Base exception for all DisasterClassifier runtime anomalies."""
    pass


class ModelLoadError(DisasterClassifierError):
    """Raised when the underlying transformer model fails to load."""
    pass


class TokenizerLoadError(DisasterClassifierError):
    """Raised when the tokenizer fails to load."""
    pass


class ClassMapLoadError(DisasterClassifierError):
    """Raised when class_map.json is missing, unreadable, or malformed."""
    pass


class InvalidInputError(DisasterClassifierError):
    """Raised when classify()/predict_batch() receives invalid input text."""
    pass


class ClassifierNotReadyError(DisasterClassifierError):
    """Raised when an inference call is attempted before the classifier has
    successfully finished initializing all of its components."""
    pass


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class ClassificationResult:
    """
    Structured result of a single classification call. Provided as a
    dataclass for type-safety and IDE autocomplete internally; `classify()`
    and `predict_batch()` return the plain-dict form (`as_dict()`) to match
    the framework's established public-API convention (see
    communication/nodes.py's `as_dict()` pattern) and to keep downstream
    consumers dependency-free from this module's internal types.
    """

    text: str
    predicted_class: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    inference_time_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted_class": self.predicted_class,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp,
            "inference_time_ms": self.inference_time_ms,
        }


# ============================================================================
# DISASTER CLASSIFIER
# ============================================================================

class DisasterClassifier:
    """
    Loads the trained multi-class disaster-message classification model
    exactly once, then exposes fast, repeatable inference through
    `classify()` and `predict_batch()`.

    Typical usage:
        classifier = DisasterClassifier()
        result = classifier.classify("Need ambulance immediately")
        print(result["predicted_class"], result["confidence"])
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        class_map_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_sequence_length: int = 128,
    ) -> None:
        """
        Loads the tokenizer, model, and class map exactly once and prepares
        the classifier for repeated inference calls.

        Args:
            model_dir: Directory containing config.json, model.safetensors,
                tokenizer.json, tokenizer_config.json. Defaults to
                `Disaster_Prediction/models/multi_class_disaster_model`.
            class_map_path: Path to class_map.json. Defaults to
                `Disaster_Prediction/models/class_map.json`.
            device: Force a specific device ("cpu" or "cuda"). If omitted,
                CUDA is used automatically when available, else CPU.
            max_sequence_length: Maximum token length used for truncation
                and padding during inference.

        Raises:
            ModelLoadError: If the model cannot be loaded.
            TokenizerLoadError: If the tokenizer cannot be loaded.
            ClassMapLoadError: If class_map.json is missing or malformed.
        """
        self._model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self._class_map_path = Path(class_map_path) if class_map_path is not None else DEFAULT_CLASS_MAP_PATH
        self._max_sequence_length = max_sequence_length

        self._device = self._resolve_device(device)
        logger.info(f"DisasterClassifier initializing -> device selected: {self._device}")

        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()
        self._id_to_label, self._label_to_id = self._load_class_map()

        self._ready = True
        logger.info(
            f"DisasterClassifier ready -> {len(self._id_to_label)} classes loaded "
            f"from '{self._class_map_path.name}'."
        )

    # ------------------------------------------------------------------------
    # INITIALIZATION HELPERS
    # ------------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """Resolves the target compute device, falling back gracefully when CUDA is unavailable."""
        if device is not None:
            requested = device.lower().strip()
            if requested == "cuda" and not torch.cuda.is_available():
                logger.warning("Requested device 'cuda' but CUDA is unavailable; falling back to CPU.")
                return torch.device("cpu")
            try:
                return torch.device(requested)
            except RuntimeError as exc:
                raise DisasterClassifierError(f"Invalid device specifier '{device}': {exc}") from exc

        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.info("CUDA unavailable -> falling back to CPU for inference.")
        return torch.device("cpu")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Loads the tokenizer from `model_dir` exactly once."""
        if not self._model_dir.exists():
            raise TokenizerLoadError(f"Model directory not found: '{self._model_dir}'.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        except Exception as exc:
            logger.error(f"Failed to load tokenizer from '{self._model_dir}': {exc}", exc_info=True)
            raise TokenizerLoadError(f"Failed to load tokenizer from '{self._model_dir}': {exc}") from exc
        logger.info(f"Tokenizer loaded successfully from '{self._model_dir}'.")
        return tokenizer

    def _load_model(self) -> AutoModelForSequenceClassification:
        """Loads the sequence-classification model exactly once, moves it to
        the resolved device, and switches it to evaluation mode."""
        if not self._model_dir.exists():
            raise ModelLoadError(f"Model directory not found: '{self._model_dir}'.")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir))
        except Exception as exc:
            logger.error(f"Failed to load model from '{self._model_dir}': {exc}", exc_info=True)
            raise ModelLoadError(f"Failed to load model from '{self._model_dir}': {exc}") from exc

        model.to(self._device)
        model.eval()
        logger.info(f"Model loaded successfully from '{self._model_dir}' and moved to '{self._device}'.")
        return model

    def _load_class_map(self) -> tuple[Dict[int, str], Dict[str, int]]:
        """
        Loads class_map.json and normalizes it into both an id->label and a
        label->id dict, regardless of which direction the file was authored
        in (`{"0": "label"}` or `{"label": 0}` are both accepted).
        """
        if not self._class_map_path.exists():
            raise ClassMapLoadError(f"class_map.json not found at '{self._class_map_path}'.")

        try:
            with open(self._class_map_path, "r", encoding="utf-8") as fh:
                raw_map = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ClassMapLoadError(f"class_map.json at '{self._class_map_path}' contains invalid JSON: {exc}") from exc
        except OSError as exc:
            raise ClassMapLoadError(f"class_map.json at '{self._class_map_path}' could not be read: {exc}") from exc

        if not isinstance(raw_map, dict) or not raw_map:
            raise ClassMapLoadError(f"class_map.json at '{self._class_map_path}' must contain a non-empty JSON object.")

        # Detect orientation: keys are numeric class ids ("0", "1", ...) vs.
        # keys are label names mapping to numeric ids.
        keys_are_numeric = all(str(k).lstrip("-").isdigit() for k in raw_map.keys())

        id_to_label: Dict[int, str] = {}
        label_to_id: Dict[str, int] = {}

        try:
            if keys_are_numeric:
                for key, value in raw_map.items():
                    class_id = int(key)
                    label = str(value)
                    id_to_label[class_id] = label
                    label_to_id[label] = class_id
            else:
                for key, value in raw_map.items():
                    label = str(key)
                    class_id = int(value)
                    id_to_label[class_id] = label
                    label_to_id[label] = class_id
        except (TypeError, ValueError) as exc:
            raise ClassMapLoadError(f"class_map.json at '{self._class_map_path}' has a malformed id/label pair: {exc}") from exc

        return id_to_label, label_to_id

    # ------------------------------------------------------------------------
    # INTERNAL INFERENCE PIPELINE
    # ------------------------------------------------------------------------

    def _validate_text(self, text: Any) -> str:
        """Validates a single input text, raising InvalidInputError on anything unusable."""
        if text is None:
            raise InvalidInputError("classify() received None; expected a non-empty string.")
        if not isinstance(text, str):
            raise InvalidInputError(f"classify() expected a string, got {type(text).__name__}.")
        if not text.strip():
            raise InvalidInputError("classify() received an empty or whitespace-only string.")
        return text

    def _run_inference_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Tokenizes and runs a batch of already-validated texts through the
        model, returning a (batch_size, num_classes) tensor of softmax
        probabilities on CPU.
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_sequence_length,
            return_tensors="pt",
        )
        encoded = {key: tensor.to(self._device) for key, tensor in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)

        return probabilities.detach().cpu()

    def _build_result(self, text: str, probability_row: torch.Tensor, inference_time_ms: float) -> ClassificationResult:
        """Converts a single row of class probabilities into a ClassificationResult."""
        probs_list = probability_row.tolist()
        predicted_id = int(probability_row.argmax().item())
        predicted_label = self._id_to_label.get(predicted_id, f"unknown_class_{predicted_id}")
        confidence = float(probs_list[predicted_id])

        probabilities: Dict[str, float] = {
            self._id_to_label.get(idx, f"unknown_class_{idx}"): round(float(p), 6)
            for idx, p in enumerate(probs_list)
        }

        return ClassificationResult(
            text=text,
            predicted_class=predicted_label,
            class_id=predicted_id,
            confidence=round(confidence, 6),
            probabilities=probabilities,
            timestamp=datetime.now(timezone.utc).isoformat(),
            inference_time_ms=round(inference_time_ms, 3),
        )

    # ------------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------------

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single disaster-related message.

        Args:
            text: The message to classify.

        Returns:
            A dict with keys: text, predicted_class, class_id, confidence,
            probabilities, timestamp, inference_time_ms.

        Raises:
            InvalidInputError: If `text` is None, empty, or not a string.
            ClassifierNotReadyError: If called before initialization completed.
            DisasterClassifierError: On unexpected inference failure.
        """
        self._assert_ready()
        validated_text = self._validate_text(text)

        logger.info(f"Prediction started -> text='{validated_text[:80]}'")
        start = time.perf_counter()

        try:
            probabilities = self._run_inference_batch([validated_text])
        except Exception as exc:
            logger.error(f"Inference failed for text='{validated_text[:80]}': {exc}", exc_info=True)
            raise DisasterClassifierError(f"Inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = self._build_result(validated_text, probabilities[0], elapsed_ms)

        logger.info(
            f"Prediction completed -> class='{result.predicted_class}' "
            f"confidence={result.confidence:.4f} inference_time_ms={result.inference_time_ms:.2f}"
        )
        return result.as_dict()

    def predict_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Classifies a batch of disaster-related messages in a single forward pass.

        Args:
            messages: A list of message strings to classify.

        Returns:
            A list of dicts, one per input message, in the same order as
            `messages`, each matching the `classify()` output schema.

        Raises:
            InvalidInputError: If `messages` is None, empty, not a list, or
                contains any invalid entries.
            ClassifierNotReadyError: If called before initialization completed.
            DisasterClassifierError: On unexpected inference failure.
        """
        self._assert_ready()

        if messages is None:
            raise InvalidInputError("predict_batch() received None; expected a list of strings.")
        if not isinstance(messages, list):
            raise InvalidInputError(f"predict_batch() expected a list, got {type(messages).__name__}.")
        if not messages:
            raise InvalidInputError("predict_batch() received an empty list.")

        validated_texts = [self._validate_text(msg) for msg in messages]

        logger.info(f"Prediction started -> batch of {len(validated_texts)} message(s)")
        start = time.perf_counter()

        try:
            probabilities = self._run_inference_batch(validated_texts)
        except Exception as exc:
            logger.error(f"Batch inference failed for {len(validated_texts)} message(s): {exc}", exc_info=True)
            raise DisasterClassifierError(f"Batch inference failed: {exc}") from exc

        total_elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_item_elapsed_ms = total_elapsed_ms / len(validated_texts)

        results = [
            self._build_result(text, probabilities[i], per_item_elapsed_ms)
            for i, text in enumerate(validated_texts)
        ]

        logger.info(
            f"Prediction completed -> batch of {len(results)} message(s) "
            f"in {total_elapsed_ms:.2f}ms total ({per_item_elapsed_ms:.2f}ms/item)"
        )
        return [result.as_dict() for result in results]

    def get_supported_classes(self) -> List[str]:
        """Returns the list of all disaster classes the model can predict, ordered by class id."""
        self._assert_ready()
        return [self._id_to_label[class_id] for class_id in sorted(self._id_to_label.keys())]

    def health_check(self) -> bool:
        """
        Verifies that the tokenizer, model, and class map are all loaded and
        that the classifier is ready to serve predictions.

        Returns:
            True if every component is loaded and ready.

        Raises:
            ClassifierNotReadyError: If any required component is missing.
        """
        missing_components: List[str] = []

        if getattr(self, "_tokenizer", None) is None:
            missing_components.append("tokenizer")
        if getattr(self, "_model", None) is None:
            missing_components.append("model")
        if not getattr(self, "_id_to_label", None):
            missing_components.append("class_map")

        if missing_components:
            raise ClassifierNotReadyError(
                f"Health check failed -> missing component(s): {', '.join(missing_components)}"
            )

        logger.info("Health check passed -> tokenizer, model, and class map are all loaded.")
        return True

    # ------------------------------------------------------------------------
    # INTERNAL GUARDS
    # ------------------------------------------------------------------------

    def _assert_ready(self) -> None:
        """Raises ClassifierNotReadyError if initialization did not complete successfully."""
        if not getattr(self, "_ready", False):
            raise ClassifierNotReadyError("DisasterClassifier is not ready; initialization did not complete.")


# ============================================================================
# DEMO ENTRY POINT
# ============================================================================

def _print_prediction(result: Dict[str, Any]) -> None:
    """Pretty-prints a single classification result to the console."""
    print(f"Text        : {result['text']}")
    print(f"Predicted   : {result['predicted_class']}  (class_id={result['class_id']})")
    print(f"Confidence  : {result['confidence']:.4f}")
    print(f"Inference   : {result['inference_time_ms']:.2f} ms")
    print(f"Timestamp   : {result['timestamp']}")
    print("Probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {label:<30s} {prob:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    classifier = DisasterClassifier()

    print(f"Health check passed: {classifier.health_check()}")
    print(f"Supported classes ({len(classifier.get_supported_classes())}):")
    for cls in classifier.get_supported_classes():
        print(f"  - {cls}")
    print("=" * 60)

    demo_messages = [
        "There are injured people trapped inside the building.",
        "We need food and drinking water.",
        "Bridge collapsed due to flood.",
    ]

    print("\nSingle classify() calls:\n")
    for message in demo_messages:
        prediction = classifier.classify(message)
        _print_prediction(prediction)

    print("\npredict_batch() call:\n")
    batch_results = classifier.predict_batch(demo_messages)
    for prediction in batch_results:
        _print_prediction(prediction)