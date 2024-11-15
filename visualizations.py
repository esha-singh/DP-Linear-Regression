import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.linear_model import LinearRegression

def generate_bounded_data(n_samples: int, slope: float, intercept: float, noise_scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in [0,1] range"""
    x = np.random.uniform(0, 1, n_samples)
    y = slope * x + intercept + np.random.normal(0, noise_scale, n_samples)
    return x, np.clip(y, 0, 1)

def plot_regression_comparison(n_samples: int = 200, epsilon: float = 1.0):
    """Plot comparison between DP and non-DP regression"""
    # Generate data
    true_slope, true_intercept = 0.5, 0.2
    x, y = generate_bounded_data(n_samples, true_slope, true_intercept)
    
    # Fit models
    dp_reg = DPBoundedLinearRegression(epsilon=epsilon, delta=1e-5)
    dp_reg.fit(x, y)
    
    non_dp_reg = LinearRegression()
    non_dp_reg.fit(x.reshape(-1, 1), y)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, alpha=0.5, label='Data points')
    
    # Plot regression lines
    x_line = np.array([0, 1])
    plt.plot(x_line, true_slope * x_line + true_intercept, 'r-', 
             label=f'True (slope={true_slope:.3f}, intercept={true_intercept:.3f})')
    plt.plot(x_line, dp_reg.slope * x_line + dp_reg.intercept, 'g--', 
             label=f'DP (slope={dp_reg.slope:.3f}, intercept={dp_reg.intercept:.3f})')
    plt.plot(x_line, non_dp_reg.coef_[0] * x_line + non_dp_reg.intercept_, 'b:', 
             label=f'Non-DP (slope={non_dp_reg.coef_[0]:.3f}, intercept={non_dp_reg.intercept_:.3f})')
    
    plt.title(f'Regression Comparison (ε={epsilon})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("regression_plot.png")
    plt.show()

def plot_privacy_utility_tradeoff(n_trials: int = 20):
    """Plot privacy-utility tradeoff with different epsilon values"""
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    true_slope, true_intercept = 0.5, 0.2
    n_samples = 1000
    
    slope_errors = []
    intercept_errors = []
    
    for epsilon in epsilons:
        slope_trial_errors = []
        intercept_trial_errors = []
        
        for _ in range(n_trials):
            x, y = generate_bounded_data(n_samples, true_slope, true_intercept)
            dp_reg = DPBoundedLinearRegression(epsilon=epsilon, delta=1e-5)
            dp_reg.fit(x, y)
            
            slope_trial_errors.append(abs(dp_reg.slope - true_slope))
            intercept_trial_errors.append(abs(dp_reg.intercept - true_intercept))
        
        slope_errors.append(np.mean(slope_trial_errors))
        intercept_errors.append(np.mean(intercept_trial_errors))
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(epsilons, slope_errors, 'o-', label='Slope Error')
    plt.semilogx(epsilons, intercept_errors, 's-', label='Intercept Error')
    plt.title('Privacy-Utility Tradeoff')
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('Average Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("privacy_utility_tradeoff.png")
    plt.show()

def plot_sample_size_impact():
    """Plot impact of sample size on estimation error"""
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    epsilons = [0.1, 1.0, 3.0, 5.0, 10.0]
    true_slope, true_intercept = 0.5, 0.2
    n_trials = 10
    
    plt.figure(figsize=(12, 6))
    
    for epsilon in epsilons:
        errors = []
        for n in sample_sizes:
            trial_errors = []
            for _ in range(n_trials):
                x, y = generate_bounded_data(n, true_slope, true_intercept)
                dp_reg = DPBoundedLinearRegression(epsilon=epsilon, delta=1e-5)
                dp_reg.fit(x, y)
                mse = np.mean((y - dp_reg.predict(x))**2)
                trial_errors.append(mse)
            errors.append(np.mean(trial_errors))
        
        plt.semilogx(sample_sizes, errors, 'o-', label=f'ε={epsilon}')
    
    plt.title('Impact of Sample Size on Estimation Error')
    plt.xlabel('Sample Size')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("sample_size_impact.png")
    plt.show()

def plot_noise_distribution(n_trials: int = 1000):
    """Plot distribution of noise added to statistics"""
    epsilon = 1.0
    n_samples = 1000
    true_slope, true_intercept = 0.5, 0.2
    
    slope_estimates = []
    intercept_estimates = []
    
    for _ in range(n_trials):
        x, y = generate_bounded_data(n_samples, true_slope, true_intercept)
        dp_reg = DPBoundedLinearRegression(epsilon=epsilon, delta=1e-5)
        dp_reg.fit(x, y)
        slope_estimates.append(dp_reg.slope)
        intercept_estimates.append(dp_reg.intercept)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot slope distribution
    ax1.hist(slope_estimates, bins=30, density=True, alpha=0.7)
    ax1.axvline(true_slope, color='r', linestyle='--', label='True Value')
    ax1.set_title('Distribution of Slope Estimates')
    ax1.set_xlabel('Slope')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    
    # Plot intercept distribution
    ax2.hist(intercept_estimates, bins=30, density=True, alpha=0.7)
    ax2.axvline(true_intercept, color='r', linestyle='--', label='True Value')
    ax2.set_title('Distribution of Intercept Estimates')
    ax2.set_xlabel('Intercept')
    ax2.legend()
    
    
    plt.tight_layout()
    plt.savefig("noise_distribution.png")
    plt.show()
    

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(40)
    
    # Generate all plots
    print("1. Regression Comparison Plot")
    plot_regression_comparison()
    
    print("\n2. Privacy-Utility Tradeoff Plot")
    plot_privacy_utility_tradeoff()
    
    print("\n3. Sample Size Impact Plot")
    plot_sample_size_impact()
    
    print("\n4. Noise Distribution Plot")
    plot_noise_distribution()
