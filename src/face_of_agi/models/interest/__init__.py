"""Interest model role."""

from face_of_agi.contracts import CandidateValuePrediction, InterestPrediction
from face_of_agi.models.interest.adapter import HFInterestAdapter, VLLMInterestAdapter
from face_of_agi.models.interest.config import HFInterestConfig, VLLMInterestConfig
from face_of_agi.models.interest.contracts import (
    InterestModel,
    InterestPredictionInput,
    interest_prediction_json_schema,
)

__all__ = [
    "CandidateValuePrediction",
    "HFInterestAdapter",
    "HFInterestConfig",
    "InterestModel",
    "InterestPrediction",
    "InterestPredictionInput",
    "VLLMInterestAdapter",
    "VLLMInterestConfig",
    "interest_prediction_json_schema",
]
