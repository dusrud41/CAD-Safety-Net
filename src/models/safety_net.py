# src/models/safety_net.py

import torch
import torch.nn.functional as F

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from config.config import *

from safety.image_quality import (
    calculate_blur_score,
    calculate_noise_score,
    calculate_brightness_score
)

from safety.ood_detector import (
    calculate_ood_score
)

from inference.uncertainty import (
    calculate_uncertainty
)


class SafetyNet:

    def __init__(self, model):

        self.model = model

    def evaluate(
        self,
        image_tensor,
        image_path
    ):

        # =================================
        # Model Prediction
        # =================================
        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        confidence = confidence.item()

        predicted_class = CLASS_NAMES[
            predicted.item()
        ]

        # =================================
        # Uncertainty
        # =================================
        uncertainty = calculate_uncertainty(
            probabilities
        )

        # =================================
        # OOD Score
        # =================================
        ood_score = calculate_ood_score(
            probabilities
        )

        # =================================
        # Image Quality
        # =================================
        blur_score = calculate_blur_score(
            image_path
        )

        noise_score = calculate_noise_score(
            image_path
        )

        brightness_score = (
            calculate_brightness_score(
                image_path
            )
        )

        # =================================
        # Decision Logic
        # =================================
        decision = "ACCEPT"

        reasons = []

        # Low Confidence
        if confidence < CONFIDENCE_THRESHOLD:

            decision = "FLAG"

            reasons.append(
                "Low confidence"
            )

        # High Uncertainty
        if uncertainty > UNCERTAINTY_THRESHOLD:

            decision = "FLAG"

            reasons.append(
                "High uncertainty"
            )

        # OOD Detection
        if ood_score > OOD_THRESHOLD:

            decision = "REJECT"

            reasons.append(
                "Out-of-distribution input"
            )

        # Noise Detection
        if noise_score > NOISE_THRESHOLD:

            decision = "REJECT"

            reasons.append(
                "Severe image noise"
            )

        # Blur Detection
        if blur_score < BLUR_THRESHOLD:

            decision = "REJECT"

            reasons.append(
                "Blurred image"
            )

        return {

            "prediction":
            predicted_class,

            "confidence":
            confidence,

            "uncertainty":
            uncertainty,

            "ood_score":
            ood_score,

            "blur_score":
            blur_score,

            "noise_score":
            noise_score,

            "brightness_score":
            brightness_score,

            "decision":
            decision,

            "reasons":
            reasons
        }