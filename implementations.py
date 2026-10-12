"""CS-433 Project 1: six required machine-learning methods (NumPy only)."""

import numpy as np


# ---------- Helpers: linear regression ----------

def compute_mse_loss(y, tx, w):
    """MSE with the 1/2 factor required by the project."""
    errors = y - tx @ w
    return 0.5 * np.mean(errors ** 2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the mean squared error."""
    errors = tx @ w - y
    return tx.T @ errors / len(y)


# ---------- 1. Linear regression: Gradient Descent ----------

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression by full-batch gradient descent."""
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w -= gamma * gradient

    return w, compute_mse_loss(y, tx, w)


# ---------- 2. Linear regression: Stochastic Gradient Descent ----------

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with one randomly sampled row per step."""
    w = np.asarray(initial_w, dtype=float).copy()
    n_samples = len(y)

    for _ in range(max_iters):
        i = np.random.randint(n_samples)
        error = tx[i] @ w - y[i]
        gradient = tx[i] * error
        w -= gamma * gradient

    return w, compute_mse_loss(y, tx, w)


# ---------- 3. Linear regression: Least Squares ----------

def least_squares(y, tx):
    """Solve the normal equations for ordinary least squares."""
    a = tx.T @ tx
    b = tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        # Handles rank-deficient design matrices.
        w = np.linalg.pinv(tx) @ y

    return w, compute_mse_loss(y, tx, w)


# ---------- 4. Linear regression: Ridge Regression ----------

def ridge_regression(y, tx, lambda_):
    """Solve ridge normal equations, regularizing all coefficients."""
    n_samples, n_features = tx.shape
    a = tx.T @ tx + 2 * n_samples * lambda_ * np.eye(n_features)
    b = tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b

    # Project specification: return the unregularized MSE.
    return w, compute_mse_loss(y, tx, w)


# ---------- Helpers: logistic regression ----------

def sigmoid(t):
    """Numerically stable sigmoid."""
    t = np.clip(t, -500, 500)
    return 1.0 / (1.0 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """Mean negative log-likelihood for labels in {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_logistic_gradient(y, tx, w):
    """Gradient of the mean logistic loss."""
    predictions = sigmoid(tx @ w)
    return tx.T @ (predictions - y) / len(y)


# ---------- 5. Logistic Regression ----------

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit logistic regression by full-batch gradient descent."""
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = calculate_logistic_gradient(y, tx, w)
        w -= gamma * gradient

    return w, calculate_logistic_loss(y, tx, w)


# ---------- 6. Regularized Logistic Regression ----------

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit L2-regularized logistic regression by gradient descent."""
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = calculate_logistic_gradient(y, tx, w)
        gradient += 2 * lambda_ * w
        w -= gamma * gradient

    # Project specification: return the unregularized logistic loss.
    return w, calculate_logistic_loss(y, tx, w)
