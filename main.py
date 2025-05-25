import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI display
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


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


def trapezoidal_membership(x, a, b, c, d):
    '''
    Trapezoidal Membership Function

    Parameters:
        x: Input value
        a: Left foot of trapezoid
        b: Left shoulder of trapezoid
        c: Right shoulder of trapezoid
        d: Right foot of trapezoid

    Returns:
        Membership degree between 0 and 1
    '''
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0.0


def sigmoid_membership(x, a, c):
    '''
    Sigmoid Membership Function

    Parameters:
        x: Input value
        a: Slope of the sigmoid
        c: Center of the sigmoid

    Returns:
        Membership degree between 0 and 1
    '''
    return 1 / (1 + np.exp(-a * (x - c)))


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
        elif func == 'trapezoidal':
            func_vectorized = np.vectorize(
                lambda x_val: trapezoidal_membership(x_val % 24, *params))
            y = func_vectorized(x_range)
        elif func == 'sigmoid':
            func_vectorized = np.vectorize(
                lambda x_val: sigmoid_membership(x_val, *params))
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


def create_fuzzy_controller():
    """
    Creates and returns a fuzzy logic controller for screen brightness
    """
    # Create input variables
    ambient_light = ctrl.Antecedent(np.arange(0, 101, 1), 'ambient_light')
    battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')
    usage_time = ctrl.Antecedent(np.arange(0, 241, 1), 'usage_time')
    time_of_day = ctrl.Antecedent(np.arange(0, 24, 1), 'time_of_day')

    # Create output variable
    brightness = ctrl.Consequent(np.arange(0, 101, 1), 'brightness')

    # Define membership functions for ambient light
    ambient_light['dark'] = fuzz.gaussmf(ambient_light.universe, 20, 10)
    ambient_light['moderate'] = fuzz.gaussmf(ambient_light.universe, 50, 10)
    ambient_light['bright'] = fuzz.gaussmf(ambient_light.universe, 80, 10)

    # Define membership functions for battery level
    battery_level['low'] = fuzz.gaussmf(battery_level.universe, 20, 10)
    battery_level['medium'] = fuzz.gaussmf(battery_level.universe, 50, 10)
    battery_level['high'] = fuzz.gaussmf(battery_level.universe, 80, 10)

    # Define membership functions for usage time
    usage_time['short'] = fuzz.sigmf(usage_time.universe, -0.1, 60)
    usage_time['moderate'] = fuzz.trimf(usage_time.universe, [60, 120, 180])
    usage_time['long'] = fuzz.sigmf(usage_time.universe, 0.1, 180)

    # Define membership functions for time of day
    time_of_day['morning'] = fuzz.trapmf(time_of_day.universe, [5, 7, 9, 11])
    time_of_day['afternoon'] = fuzz.trapmf(time_of_day.universe, [11, 13, 15, 17])
    time_of_day['evening'] = fuzz.trapmf(time_of_day.universe, [17, 19, 20, 22])
    time_of_day['night'] = fuzz.trapmf(time_of_day.universe, [21, 22, 23, 23.99])
    time_of_day['early_morning'] = fuzz.trapmf(time_of_day.universe, [0, 0, 3, 5])

    # Define membership functions for brightness
    brightness['very_low'] = fuzz.trimf(brightness.universe, [0, 0, 25])
    brightness['low'] = fuzz.trimf(brightness.universe, [0, 25, 50])
    brightness['medium'] = fuzz.trimf(brightness.universe, [25, 50, 75])
    brightness['high'] = fuzz.trimf(brightness.universe, [50, 75, 100])
    brightness['very_high'] = fuzz.trimf(brightness.universe, [75, 100, 100])

    # Create rules
    rule1 = ctrl.Rule(
        ambient_light['dark'] & battery_level['high'] & usage_time['short'],
        brightness['medium']
    )
    rule2 = ctrl.Rule(
        ambient_light['dark'] & battery_level['low'] & usage_time['long'],
        brightness['low']
    )
    rule3 = ctrl.Rule(
        ambient_light['bright'] & battery_level['high'],
        brightness['high']
    )
    rule4 = ctrl.Rule(
        ambient_light['moderate'] & (time_of_day['night'] | time_of_day['early_morning']),
        brightness['low']
    )
    rule5 = ctrl.Rule(
        ambient_light['bright'] & time_of_day['morning'],
        brightness['very_high']
    )
    rule6 = ctrl.Rule(
        battery_level['low'] & usage_time['long'],
        brightness['very_low']
    )

    # Create control system
    brightness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    brightness_sim = ctrl.ControlSystemSimulation(brightness_ctrl)

    return brightness_sim

def get_brightness_level(ambient_light, battery_level, usage_time, time_of_day):
    """
    Calculate the appropriate brightness level based on input conditions
    
    Parameters:
        ambient_light: Ambient light level (0-100)
        battery_level: Battery level (0-100)
        usage_time: Screen usage time in minutes (0-240)
        time_of_day: Hour of the day (0-23)
    
    Returns:
        Recommended brightness level (0-100)
    """
    sim = create_fuzzy_controller()
    
    # Input values
    sim.input['ambient_light'] = ambient_light
    sim.input['battery_level'] = battery_level
    sim.input['usage_time'] = usage_time
    sim.input['time_of_day'] = time_of_day
    
    # Compute result
    sim.compute()
    
    return sim.output['brightness']

def draw_memberships():
    """
    Draws the membership functions used in the project.
    """
    # Create input variables
    ambient_light = ctrl.Antecedent(np.arange(0, 101, 1), 'ambient_light')
    battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')
    usage_time = ctrl.Antecedent(np.arange(0, 241, 1), 'usage_time')
    time_of_day = ctrl.Antecedent(np.arange(0, 24, 1), 'time_of_day')
    brightness = ctrl.Consequent(np.arange(0, 101, 1), 'brightness')

    # Define membership functions for ambient light
    ambient_light['dark'] = fuzz.gaussmf(ambient_light.universe, 20, 10)
    ambient_light['moderate'] = fuzz.gaussmf(ambient_light.universe, 50, 10)
    ambient_light['bright'] = fuzz.gaussmf(ambient_light.universe, 80, 10)

    # Define membership functions for battery level
    battery_level['low'] = fuzz.gaussmf(battery_level.universe, 20, 10)
    battery_level['medium'] = fuzz.gaussmf(battery_level.universe, 50, 10)
    battery_level['high'] = fuzz.gaussmf(battery_level.universe, 80, 10)

    # Define membership functions for usage time
    usage_time['short'] = fuzz.sigmf(usage_time.universe, -0.1, 60)
    usage_time['moderate'] = fuzz.trimf(usage_time.universe, [60, 120, 180])
    usage_time['long'] = fuzz.sigmf(usage_time.universe, 0.1, 180)

    # Define membership functions for time of day
    time_of_day['morning'] = fuzz.trapmf(time_of_day.universe, [5, 7, 9, 11])
    time_of_day['afternoon'] = fuzz.trapmf(time_of_day.universe, [11, 13, 15, 17])
    time_of_day['evening'] = fuzz.trapmf(time_of_day.universe, [17, 19, 20, 22])
    time_of_day['night'] = fuzz.trapmf(time_of_day.universe, [21, 22, 23, 23.99])
    time_of_day['early_morning'] = fuzz.trapmf(time_of_day.universe, [0, 0, 3, 5])

    # Define membership functions for brightness
    brightness['very_low'] = fuzz.trimf(brightness.universe, [0, 0, 25])
    brightness['low'] = fuzz.trimf(brightness.universe, [0, 25, 50])
    brightness['medium'] = fuzz.trimf(brightness.universe, [25, 50, 75])
    brightness['high'] = fuzz.trimf(brightness.universe, [50, 75, 100])
    brightness['very_high'] = fuzz.trimf(brightness.universe, [75, 100, 100])

    # Draw membership functions
    ambient_light.view()
    battery_level.view()
    usage_time.view()
    time_of_day.view()
    brightness.view()
    
    # Keep the plots open
    plt.show(block=True)

# Example usage
if __name__ == "__main__":
    # Example scenarios
    scenarios = [
        {
            'name': 'Dark room, high battery, short usage',
            'ambient_light': 20,
            'battery_level': 90,
            'usage_time': 30,
            'time_of_day': 14
        },
        {
            'name': 'Bright room, low battery, long usage',
            'ambient_light': 80,
            'battery_level': 20,
            'usage_time': 180,
            'time_of_day': 20
        }
    ]
    
    for scenario in scenarios:
        brightness = get_brightness_level(
            scenario['ambient_light'],
            scenario['battery_level'],
            scenario['usage_time'],
            scenario['time_of_day']
        )
        print(f"\nScenario: {scenario['name']}")
        print(f"Input conditions:")
        print(f"  Ambient Light: {scenario['ambient_light']}")
        print(f"  Battery Level: {scenario['battery_level']}")
        print(f"  Usage Time: {scenario['usage_time']} minutes")
        print(f"  Time of Day: {scenario['time_of_day']} hours")
        print(f"Recommended Brightness: {brightness:.1f}")

    # Draw membership functions
    print("Drawing membership functions...")
    draw_memberships()
    
