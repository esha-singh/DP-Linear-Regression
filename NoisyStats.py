import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BoundedNoisyStatsResult:
    """Results from noisy sufficient statistics computation for [0,1] bounded data"""
    slope: float
    intercept: float
    privacy_params: dict
    noisy_cov: float = None
    noisy_var: float = None

def compute_bounded_noisy_stats(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: float
) -> BoundedNoisyStatsResult:
    """
    Compute differentially private statistics for linear regression with [0,1] bounded data.
    Uses correct sensitivity for intercept based on slope estimator.
    
    Args:
        x: Input features in [0,1]
        y: Target values in [0,1]
        epsilon: Privacy parameter
        delta: Privacy parameter
    
    Returns:
        BoundedNoisyStatsResult containing private slope and intercept
    """
    # Validate inputs
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if epsilon <= 0 or delta <= 0:
        raise ValueError("Privacy parameters must be positive")
    
    # Validate bounds
    if not (np.all(x >= 0) and np.all(x <= 1) and np.all(y >= 0) and np.all(y <= 1)):
        raise ValueError("All data must be in [0,1] range")
    
    n = len(x)
    
    # Compute raw statistics
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Center the data
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Compute covariance and variance
    covariance = np.sum(x_centered * y_centered) / n
    variance = np.sum(x_centered ** 2) / n
    
    # Sensitivity for covariance and variance is (1 - 1/n)
    sensitivity_stats = 1 - 1/n # GS upper bound as in proof.
    
    # Split epsilon into thirds
    epsilon_per_stat = epsilon / 3 
    
    # Add noise to covariance and variance
    noisy_covariance = covariance + np.random.laplace(0, sensitivity_stats/epsilon_per_stat)
    noisy_variance = variance + np.random.laplace(0, sensitivity_stats/epsilon_per_stat)
    
    # Compute slope (protect against division by zero)
    slope = noisy_covariance / max(noisy_variance, 1e-10)
    
    # Compute intercept sensitivity: 1/n * (1 - max(slope))
    # Note: For [0,1] bounded data, theoretical max slope can't exceed 1
    slope_bound = abs(slope) 
    sensitivity_intercept = (1/n) * (1 + abs(slope_bound)) # GS upper bound as in proof.
    
    # Compute intercept with remaining privacy budget and correct sensitivity
    intercept = y_mean - slope * x_mean + np.random.laplace(0, sensitivity_intercept/epsilon_per_stat)
    
    # Ensure output bounds are respected
    intercept = np.clip(intercept, 0, 1)
    
    return BoundedNoisyStatsResult(
        slope=slope,
        intercept=intercept,
        privacy_params={"epsilon": epsilon, "delta": delta},
        noisy_cov=noisy_covariance,
        noisy_var=noisy_variance
    )

class DPBoundedLinearRegression:
    """Differentially private linear regression for [0,1] bounded data"""
    
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        self.slope = None
        self.intercept = None
        self.stats = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'DPBoundedLinearRegression':
        """Fit the regression model using noisy statistics"""
        # Reshape inputs if needed
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        
        # Compute noisy statistics
        self.stats = compute_bounded_noisy_stats(
            x, y,
            self.epsilon,
            self.delta
        )
        
        self.slope = self.stats.slope
        self.intercept = self.stats.intercept
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be fitted before making predictions")
        
        x = np.asarray(x).reshape(-1)
        predictions = self.slope * x + self.intercept
        
        # Clip predictions to [0,1] since we know y is bounded
        return np.clip(predictions, 0, 1)

# Example usage and testing
def test_bounded_noisy_stats():
    # Generate sample data in [0,1]
    np.random.seed(40)
    n_samples = 1000
    
    # Generate x uniformly in [0,1]
    x = np.random.uniform(0, 1, n_samples)
    
    # Generate y = ax + b + noise, then clip to [0,1]
    true_slope = 0.5
    true_intercept = 0.2
    y = true_slope * x + true_intercept + np.random.normal(0, 0.05, n_samples)
    y = np.clip(y, 0, 1)
    
    # Create and fit model
    dp_reg = DPBoundedLinearRegression(epsilon=1.0, delta=1e-5)
    dp_reg.fit(x, y)
    
    # Print results
    print(f"True slope: {true_slope:.3f}, Estimated slope: {dp_reg.slope:.3f}")
    print(f"True intercept: {true_intercept:.3f}, Estimated intercept: {dp_reg.intercept:.3f}")
    
    # Print sensitivities
    n = len(x)
    sensitivity_stats = 1 - 1/n
    sensitivity_intercept = (1/n) * (1 + abs(dp_reg.slope))
    print(f"\nCovariance/Variance sensitivity: {sensitivity_stats:.6f}")
    print(f"Intercept sensitivity: {sensitivity_intercept:.6f}")
    
    # Print noisy statistics
    print(f"Noisy covariance: {dp_reg.stats.noisy_cov:.6f}")
    print(f"Noisy variance: {dp_reg.stats.noisy_var:.6f}")
    
    # Compute and print mean squared error
    y_pred = dp_reg.predict(x)
    mse = np.mean((y - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

if __name__ == "__main__":
    test_bounded_noisy_stats()
