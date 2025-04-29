import numpy as np
import matplotlib.pyplot as plt


def gaussian_membership(x, mean, sigma):
    '''
    Gaussian Membership Function / Normal Distribution

    Parameters:
        x: Input value
        mean: Center of the distribution
        sigma: Width parameter of the distribution

    Returns:
        Membership degree between 0 and 1
    '''
    result = np.exp(-((x - mean)**2) / (sigma**2))
    return result

# Create a triangular membership function


def triangular_membership(x, a, b, c):
    '''
    Triangular Membership Function

    Parameters:
        x: Input value
        a: Left vertex of the triangle
        b: Center vertex of the triangle
        c: Right vertex of the triangle

    Returns:
        Membership degree between 0 and 1
    '''
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 0.0


def plot_membership(x_range, sets, title, xlabel):
    '''
    Plots multiple membership functions

    Parameters:
        x_range: Range of x values (numpy array)
        sets: Dictionary {label: (function, parameters)}
        title: Plot title
        xlabel: X-axis label
    '''
    plt.figure(figsize=(8, 5))
    for label, (func, params) in sets.items():
        if func == 'triangular':
            func_vectorized = np.vectorize(
                lambda x_val: triangular_membership(x_val, *params))
            y = func_vectorized(x_range)
        elif func == 'gaussian':
            func_vectorized = np.vectorize(
                lambda x_val: gaussian_membership(x_val, *params))
            y = func_vectorized(x_range)
        else:
            raise ValueError(f"Unsupported function type: {func}")

        plt.plot(x_range, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example: Ambient Light Level using Gaussian Membership
    x_ambient = np.linspace(0, 100, 1000)
    ambient_sets = {
        'Dark': ('gaussian', [20, 10]),
        'Moderate': ('gaussian', [50, 10]),
        'Bright': ('gaussian', [80, 10])
    }
    plot_membership(x_ambient, ambient_sets,
                    'Ambient Light Level', 'Ambient Light (0-100)')

    # Example: Battery Level using Gaussian Membership
    x_battery = np.linspace(0, 100, 1000)
    battery_sets = {
        'Low': ('gaussian', [20, 10]),
        'Medium': ('gaussian', [50, 10]),
        'High': ('gaussian', [80, 10])
    }
    plot_membership(x_battery, battery_sets,
                    'Battery Level', 'Battery Level (0-100)')

    # Example: Screen Usage Duration using Gaussian Membership
    x_usage = np.linspace(0, 240, 1000)
    usage_sets = {
        'Short': ('gaussian', [30, 15]),
        'Moderate': ('gaussian', [90, 30]),
        'Long': ('gaussian', [180, 30])
    }
    plot_membership(x_usage, usage_sets, 'Screen Usage Duration',
                    'Usage Duration (minutes)')

    # Example: Time of Day using Gaussian Membership
    x_time = np.linspace(0, 24, 1000)
    time_sets = {
        'Morning': ('gaussian', [8, 2]),
        'Afternoon': ('gaussian', [14, 2]),
        'Evening': ('gaussian', [19, 2]),
        'Night': ('gaussian', [1, 3])  # Night approximated centered at 1AM
    }
    plot_membership(x_time, time_sets, 'Time of Day', 'Time (Hours)')
