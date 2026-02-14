"""
Wrapper functions for converting inference results to API schemas.

This module provides conversion utilities between aerovision_inference
output formats and the API response schemas.
"""

from typing import Any

from app.schemas.aircraft import AircraftResult, Prediction
from app.schemas.airline import AirlineResult
from app.schemas.quality import QualityDetails, QualityResult
from app.schemas.registration import OcrMatch, RegistrationResult, YoloBox


def wrap_quality_result(inference_result: dict[str, Any]) -> QualityResult:
    """
    Convert inference quality result to API schema.

    Args:
        inference_result: Raw result from QualityAssessor.assess()

    Returns:
        QualityResult: API-formatted quality result
    """
    details = inference_result.get("details", {})

    return QualityResult.model_validate({
        "pass": inference_result.get("pass", False),
        "score": inference_result.get("score", 0.0),
        "details": {
            "sharpness": details.get("sharpness", 0.0),
            "exposure": details.get("exposure", 0.0),
            "composition": details.get("composition", 0.0),
            "noise": details.get("noise", 0.0),
            "color": details.get("color", 0.0)
        }
    })


def wrap_aircraft_result(inference_result: dict[str, Any]) -> AircraftResult:
    """
    Convert inference aircraft result to API schema.

    Args:
        inference_result: Raw result from AircraftClassifier.predict()

    Returns:
        AircraftResult: API-formatted aircraft result
    """
    predictions = inference_result.get("predictions", [])
    top1 = inference_result.get("top1", predictions[0] if predictions else {})

    wrapped_predictions = [
        {"class": p.get("class", ""), "confidence": p.get("confidence", 0.0)}
        for p in predictions
    ]

    return AircraftResult.model_validate({
        "top1": {
            "class": top1.get("class", ""),
            "confidence": top1.get("confidence", 0.0)
        },
        "top_k": inference_result.get("top_k", len(predictions)),
        "predictions": wrapped_predictions
    })


def wrap_airline_result(inference_result: dict[str, Any]) -> AirlineResult:
    """
    Convert inference airline result to API schema.

    Args:
        inference_result: Raw result from AirlineClassifier.predict()

    Returns:
        AirlineResult: API-formatted airline result
    """
    predictions = inference_result.get("predictions", [])
    top1 = inference_result.get("top1", predictions[0] if predictions else {})

    wrapped_predictions = [
        {"class": p.get("class", ""), "confidence": p.get("confidence", 0.0)}
        for p in predictions
    ]

    return AirlineResult.model_validate({
        "top1": {
            "class": top1.get("class", ""),
            "confidence": top1.get("confidence", 0.0)
        },
        "top_k": inference_result.get("top_k", len(predictions)),
        "predictions": wrapped_predictions
    })


def wrap_registration_result(inference_result: dict[str, Any]) -> RegistrationResult:
    """
    Convert inference registration OCR result to API schema.

    Args:
        inference_result: Raw result from RegistrationOCR.recognize()

    Returns:
        RegistrationResult: API-formatted registration result
    """
    yolo_boxes = inference_result.get("yolo_boxes", [])
    all_matches = inference_result.get("all_matches", [])

    wrapped_boxes = [
        {
            "class_id": box.get("class_id", 0),
            "x_center": box.get("x_center", 0.0),
            "y_center": box.get("y_center", 0.0),
            "width": box.get("width", 0.0),
            "height": box.get("height", 0.0),
            "text": box.get("text", ""),
            "confidence": box.get("confidence", 0.0)
        }
        for box in yolo_boxes
    ]

    wrapped_matches = [
        {
            "text": m.get("text", ""),
            "confidence": m.get("confidence", 0.0)
        }
        for m in all_matches
    ]

    return RegistrationResult.model_validate({
        "registration": inference_result.get("registration", ""),
        "confidence": inference_result.get("confidence", 0.0),
        "raw_text": inference_result.get("raw_text", ""),
        "all_matches": wrapped_matches,
        "yolo_boxes": wrapped_boxes
    })
