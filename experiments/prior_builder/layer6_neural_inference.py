"""Compact probabilistic neural inference for Layer 6 late response."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LAYER6_NEURAL_VERSION = (
    "oncotwin_layer6_probabilistic_mlp_v0_1"
)


@dataclass(frozen=True)
class Layer6NeuralConfig:
    hidden_dim: int = 8
    dropout: float = 0.0
    weight_decay: float = 0.01
    learning_rate: float = 0.01
    max_epochs: int = 300
    patience: int = 30
    min_epochs: int = 40

    def validate(self) -> None:
        if self.hidden_dim < 2:
            raise ValueError(
                "hidden_dim must be at least 2"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be nonnegative"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.max_epochs < 1:
            raise ValueError(
                "max_epochs must be positive"
            )

        if self.patience < 1:
            raise ValueError(
                "patience must be positive"
            )

        if self.min_epochs < 1:
            raise ValueError(
                "min_epochs must be positive"
            )


class Layer6ProbabilisticMLP(nn.Module):
    """Small residual MLP returning Gaussian mean and standard deviation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.linear_mean = nn.Linear(
            input_dim,
            1,
        )

        self.hidden = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.nonlinear_mean = nn.Linear(
            hidden_dim,
            1,
        )

        self.log_sd_adjustment = nn.Linear(
            hidden_dim,
            1,
        )

        self.global_log_sd = nn.Parameter(
            torch.zeros(1)
        )

        # Begin close to a regularized linear model. The nonlinear term must
        # earn its contribution during training.
        nn.init.zeros_(
            self.nonlinear_mean.weight
        )
        nn.init.zeros_(
            self.nonlinear_mean.bias
        )

        nn.init.zeros_(
            self.log_sd_adjustment.weight
        )
        nn.init.zeros_(
            self.log_sd_adjustment.bias
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(features)

        mean = (
            self.linear_mean(features)
            + self.nonlinear_mean(hidden)
        )

        log_sd = (
            self.global_log_sd
            + 0.50
            * torch.tanh(
                self.log_sd_adjustment(
                    hidden
                )
            )
        )

        log_sd = torch.clamp(
            log_sd,
            min=-4.0,
            max=2.0,
        )

        return (
            mean.squeeze(-1),
            log_sd.squeeze(-1),
        )


@dataclass
class FittedLayer6NeuralModel:
    model: Layer6ProbabilisticMLP
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    target_mean: float
    target_sd: float
    best_epoch: int
    best_validation_loss: float | None
    config: Layer6NeuralConfig
    seed: int


def _as_feature_matrix(
    features: Sequence[Sequence[float]]
    | np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        features,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix"
        )

    if matrix.shape[0] < 1:
        raise ValueError(
            "features must contain at least one row"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "features contain non-finite values"
        )

    return matrix


def _as_target_vector(
    targets: Sequence[float] | np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        targets,
        dtype=np.float32,
    ).reshape(-1)

    if vector.size < 1:
        raise ValueError(
            "targets must contain at least one value"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "targets contain non-finite values"
        )

    return vector


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _standardization(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    feature_mean = features.mean(
        axis=0
    ).astype(np.float32)

    feature_sd = features.std(
        axis=0
    ).astype(np.float32)

    feature_sd = np.where(
        feature_sd < 1e-6,
        1.0,
        feature_sd,
    ).astype(np.float32)

    target_mean = float(
        targets.mean()
    )
    target_sd = float(
        targets.std()
    )

    if target_sd < 1e-6:
        target_sd = 1.0

    return (
        feature_mean,
        feature_sd,
        target_mean,
        target_sd,
    )


def _loss(
    mean: torch.Tensor,
    log_sd: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    squared_standardized_error = (
        (target - mean) ** 2
        * torch.exp(-2.0 * log_sd)
    )

    gaussian_nll = 0.5 * (
        squared_standardized_error
        + 2.0 * log_sd
    )

    mean_stabilizer = F.smooth_l1_loss(
        mean,
        target,
        reduction="none",
    )

    return torch.mean(
        gaussian_nll
        + 0.05 * mean_stabilizer
    )


def fit_layer6_neural_model(
    train_features: Sequence[
        Sequence[float]
    ]
    | np.ndarray,
    train_targets: Sequence[float]
    | np.ndarray,
    *,
    config: Layer6NeuralConfig,
    seed: int,
    validation_features: Sequence[
        Sequence[float]
    ]
    | np.ndarray
    | None = None,
    validation_targets: Sequence[float]
    | np.ndarray
    | None = None,
    fixed_epochs: int | None = None,
) -> FittedLayer6NeuralModel:
    """Fit one probabilistic MLP using training-only normalization."""

    config.validate()
    _set_seed(seed)

    x_train = _as_feature_matrix(
        train_features
    )
    y_train = _as_target_vector(
        train_targets
    )

    if len(x_train) != len(y_train):
        raise ValueError(
            "train_features and train_targets "
            "must have equal row counts"
        )

    feature_mean, feature_sd, target_mean, target_sd = (
        _standardization(
            x_train,
            y_train,
        )
    )

    standardized_x_train = (
        x_train - feature_mean
    ) / feature_sd

    standardized_y_train = (
        y_train - target_mean
    ) / target_sd

    train_x_tensor = torch.from_numpy(
        standardized_x_train
    )
    train_y_tensor = torch.from_numpy(
        standardized_y_train
    )

    has_validation = (
        validation_features is not None
        and validation_targets is not None
        and fixed_epochs is None
    )

    if has_validation:
        x_validation = _as_feature_matrix(
            validation_features
        )
        y_validation = _as_target_vector(
            validation_targets
        )

        if len(x_validation) != len(y_validation):
            raise ValueError(
                "validation features and targets "
                "must have equal row counts"
            )

        validation_x_tensor = torch.from_numpy(
            (
                x_validation - feature_mean
            ) / feature_sd
        )

        validation_y_tensor = torch.from_numpy(
            (
                y_validation - target_mean
            ) / target_sd
        )

    model = Layer6ProbabilisticMLP(
        input_dim=x_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epochs_to_run = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else config.max_epochs
    )

    if epochs_to_run < 1:
        raise ValueError(
            "fixed_epochs must be positive"
        )

    best_state = deepcopy(
        model.state_dict()
    )
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch_index in range(
        epochs_to_run
    ):
        model.train()
        optimizer.zero_grad()

        mean, log_sd = model(
            train_x_tensor
        )

        training_loss = _loss(
            mean,
            log_sd,
            train_y_tensor,
        )

        training_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        epoch = epoch_index + 1

        if not has_validation:
            best_state = deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            continue

        model.eval()

        with torch.no_grad():
            validation_mean, validation_log_sd = (
                model(validation_x_tensor)
            )

            validation_loss = float(
                _loss(
                    validation_mean,
                    validation_log_sd,
                    validation_y_tensor,
                ).item()
            )

        if (
            validation_loss
            < best_validation_loss - 1e-6
        ):
            best_validation_loss = (
                validation_loss
            )
            best_state = deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.min_epochs
            and epochs_without_improvement
            >= config.patience
        ):
            break

    model.load_state_dict(best_state)
    model.eval()

    return FittedLayer6NeuralModel(
        model=model,
        feature_mean=feature_mean,
        feature_sd=feature_sd,
        target_mean=target_mean,
        target_sd=target_sd,
        best_epoch=max(best_epoch, 1),
        best_validation_loss=(
            None
            if not has_validation
            else float(best_validation_loss)
        ),
        config=config,
        seed=int(seed),
    )


def predict_layer6_distribution(
    fitted: FittedLayer6NeuralModel,
    features: Sequence[
        Sequence[float]
    ]
    | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive mean and standard deviation in target units."""

    matrix = _as_feature_matrix(
        features
    )

    standardized = (
        matrix - fitted.feature_mean
    ) / fitted.feature_sd

    tensor = torch.from_numpy(
        standardized.astype(np.float32)
    )

    fitted.model.eval()

    with torch.no_grad():
        standardized_mean, log_sd = (
            fitted.model(tensor)
        )

    mean = (
        fitted.target_mean
        + fitted.target_sd
        * standardized_mean.numpy()
    )

    sd = (
        fitted.target_sd
        * np.exp(log_sd.numpy())
    )

    return (
        mean.astype(float),
        np.maximum(
            sd.astype(float),
            1e-8,
        ),
    )


def combine_gaussian_ensemble(
    means: Sequence[np.ndarray],
    standard_deviations: Sequence[
        np.ndarray
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match an equally weighted Gaussian ensemble."""

    mean_matrix = np.asarray(
        means,
        dtype=float,
    )

    sd_matrix = np.asarray(
        standard_deviations,
        dtype=float,
    )

    if mean_matrix.ndim != 2:
        raise ValueError(
            "means must have shape "
            "(models, cases)"
        )

    if sd_matrix.shape != mean_matrix.shape:
        raise ValueError(
            "standard deviations must match "
            "the means shape"
        )

    if np.any(sd_matrix < 0):
        raise ValueError(
            "standard deviations must be "
            "nonnegative"
        )

    combined_mean = mean_matrix.mean(
        axis=0
    )

    second_moment = np.mean(
        sd_matrix ** 2
        + mean_matrix ** 2,
        axis=0,
    )

    combined_variance = np.maximum(
        second_moment
        - combined_mean ** 2,
        1e-12,
    )

    return (
        combined_mean,
        np.sqrt(combined_variance),
    )
