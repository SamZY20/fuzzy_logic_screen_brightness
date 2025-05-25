import numpy as np
import tkinter as tk
from tkinter import ttk  # For themed widgets like sliders
import matplotlib.pyplot as plt  # Still useful for visualizing MFs if uncommented


# --- Manual Membership Functions ---
def triangular_membership(x, a, b, c):
    """
    Manual Triangular Membership Function.
    Args:
        x (float or np.array): Input value(s).
        a (float): Left foot of the triangle.
        b (float): Peak of the triangle (membership 1.0).
        c (float): Right foot of the triangle.
    Returns:
        float or np.array: Membership degree(s) between 0 and 1.
    """
    x = np.asarray(x)  # Ensure x is a numpy array for vectorized operations

    # Initialize output array with zeros
    y = np.zeros_like(x, dtype=float)

    # Calculate membership for values between a and b
    mask_ab = (x >= a) & (x <= b)
    y[mask_ab] = (x[mask_ab] - a) / (b - a) if (b - a) != 0 else 1.0

    # Calculate membership for values between b and c
    mask_bc = (x > b) & (x <= c)
    y[mask_bc] = (c - x[mask_bc]) / (c - b) if (c - b) != 0 else 1.0

    # Handle cases where b == a or b == c (single point or step)
    if b == a and b == c:  # Single point at b
        y[x == b] = 1.0
    elif b == a:  # Step function from b to c
        mask_bc = (x >= b) & (x <= c)
        y[mask_bc] = (c - x[mask_bc]) / (c - b) if (c - b) != 0 else 1.0
    elif b == c:  # Step function from a to b
        mask_ab = (x >= a) & (x <= b)
        y[mask_ab] = (x[mask_ab] - a) / (b - a) if (b - a) != 0 else 1.0

    return y


def trapezoidal_membership(x, a, b, c, d):
    """
    Manual Trapezoidal Membership Function.
    Args:
        x (float or np.array): Input value(s).
        a (float): Left foot of trapezoid.
        b (float): Left shoulder of trapezoid (membership 1.0 starts).
        c (float): Right shoulder of trapezoid (membership 1.0 ends).
        d (float): Right foot of trapezoid.
    Returns:
        float or np.array: Membership degree(s) between 0 and 1.
    """
    x = np.asarray(x)  # Ensure x is a numpy array for vectorized operations
    y = np.zeros_like(x, dtype=float)

    # Region 1: a to b (rising slope)
    mask1 = (x >= a) & (x < b)
    y[mask1] = (x[mask1] - a) / (b - a) if (b - a) != 0 else 1.0

    # Region 2: b to c (plateau at 1.0)
    mask2 = (x >= b) & (x <= c)
    y[mask2] = 1.0

    # Region 3: c to d (falling slope)
    mask3 = (x > c) & (x <= d)
    y[mask3] = (d - x[mask3]) / (d - c) if (d - c) != 0 else 1.0

    # Handle edge cases for plateaus that are points
    if b == a:  # If left foot and shoulder are the same, it's a triangle or step
        mask1 = (x >= a) & (x <= b)
        y[mask1] = 1.0  # Treat as 1.0 at 'a' if it's the start
    if d == c:  # If right foot and shoulder are the same
        mask3 = (x >= c) & (x <= d)
        y[mask3] = 1.0  # Treat as 1.0 at 'd' if it's the end

    return y


# --- Fuzzy Logic System Class (Manual) ---
class ManualFuzzyController:
    def __init__(self):
        # Define universes for inputs and output
        self.ambient_light_universe = np.arange(0, 101, 1)
        self.battery_level_universe = np.arange(0, 101, 1)
        self.time_of_day_universe = np.arange(0, 24, 1)
        # Brightness output universe is now in Nits
        self.brightness_universe = np.arange(0, 1501, 1)  # Max 1500 nits for modern phones

        # Define membership functions for inputs
        self.ambient_light_mfs = {
            'dark': triangular_membership(self.ambient_light_universe, 0, 0, 100),
            'bright': triangular_membership(self.ambient_light_universe, 0, 100, 100)
        }
        self.battery_level_mfs = {
            'empty': triangular_membership(self.battery_level_universe, 0, 0, 100),
            'full': triangular_membership(self.battery_level_universe, 0, 100, 100)
        }
        # Night MF is a union of two trapezoids
        night_mf_part1 = trapezoidal_membership(self.time_of_day_universe, 0, 0, 6, 8)
        night_mf_part2 = trapezoidal_membership(self.time_of_day_universe, 19, 21, 23.99, 23.99)
        self.time_of_day_mfs = {
            'day': trapezoidal_membership(self.time_of_day_universe, 7, 9, 17, 19),
            'night': np.fmax(night_mf_part1, night_mf_part2)  # Union (OR) of two fuzzy sets
        }

        # Define membership functions for brightness (output) in Nits
        self.brightness_mfs = {
            'low': triangular_membership(self.brightness_universe, 0, 0, 400),  # 0-400 nits
            'medium': triangular_membership(self.brightness_universe, 200, 600, 1000),  # 200-1000 nits, peak at 600
            'high': triangular_membership(self.brightness_universe, 800, 1200, 1200)  # 800-1200 nits, peak at 1200
        }

        # Define rules (antecedent_mfs, consequent_mf_name)
        # Each rule is a tuple: ( {input_var_name: mf_name, ...}, output_mf_name )
        # Consequents now refer to nit-based brightness categories.
        self.rules = [
            # Ambient Light rules (dominant)
            ({'ambient_light': 'dark'}, 'low'),
            ({'ambient_light': 'bright'}, 'high'),

            # Battery Level rules (less dominant direct impact)
            ({'battery_level': 'empty'}, 'low'),  # If empty, still low nits target
            ({'battery_level': 'full'}, 'medium'),  # If full, medium nits target (ambient light will push it higher)

            # Time of Day rules
            ({'time_of_day': 'day'}, 'medium'),  # Day, medium nits target
            ({'time_of_day': 'night'}, 'low'),  # Night, low nits target

            # Combined rules emphasizing ambient light
            # If it's very dark, brightness should be low regardless of other factors
            ({'ambient_light': 'dark', 'battery_level': 'empty'}, 'low'),
            ({'ambient_light': 'dark', 'time_of_day': 'night'}, 'low'),
            ({'ambient_light': 'dark', 'battery_level': 'full', 'time_of_day': 'day'}, 'medium'),

            # If it's very bright, brightness should be high regardless
            ({'ambient_light': 'bright', 'battery_level': 'full'}, 'high'),
            ({'ambient_light': 'bright', 'time_of_day': 'day'}, 'high'),
            ({'ambient_light': 'bright', 'battery_level': 'empty', 'time_of_day': 'night'}, 'low'),

            # Rules to ensure battery level doesn't completely dictate brightness
            ({'ambient_light': 'bright', 'battery_level': 'empty'}, 'medium'),
            ({'ambient_light': 'dark', 'battery_level': 'full'}, 'low'),
        ]

    def fuzzify(self, input_value, universe, mfs):
        """
        Fuzzifies a crisp input value by finding its membership degree in each fuzzy set.
        Args:
            input_value (float): The crisp input.
            universe (np.array): The universe of discourse for the input variable.
            mfs (dict): Dictionary of membership functions for this variable.
        Returns:
            dict: {mf_name: membership_degree}
        """
        memberships = {}
        for name, mf_array in mfs.items():
            # Find the membership degree for the input_value
            # We need to find the index in the universe closest to input_value
            idx = np.argmin(np.abs(universe - input_value))
            memberships[name] = mf_array[idx]
        return memberships

    def infer(self, inputs):
        """
        Performs fuzzy inference (rule evaluation and implication).
        Args:
            inputs (dict): {'ambient_light': value, 'battery_level': value, 'time_of_day': value}
        Returns:
            dict: {output_mf_name: clipped_mf_array} for each activated output MF.
        """
        activated_consequents = {}  # Stores the max activation for each output MF

        for rule in self.rules:
            antecedent_conditions = rule[0]
            consequent_mf_name = rule[1]

            # Calculate rule strength (alpha-cut)
            rule_strength = 1.0  # Start with full strength

            for input_var, mf_name in antecedent_conditions.items():
                if input_var == 'ambient_light':
                    universe = self.ambient_light_universe
                    mfs = self.ambient_light_mfs
                elif input_var == 'battery_level':
                    universe = self.battery_level_universe
                    mfs = self.battery_level_mfs
                elif input_var == 'time_of_day':
                    universe = self.time_of_day_universe
                    mfs = self.time_of_day_mfs
                else:
                    raise ValueError(f"Unknown input variable: {input_var}")

                # Get the membership degree of the crisp input in the specified fuzzy set
                idx = np.argmin(np.abs(universe - inputs[input_var]))
                membership_degree = mfs[mf_name][idx]

                rule_strength = min(rule_strength, membership_degree)  # AND operation (min)

            # Implication (Mamdani: clip the consequent MF)
            clipped_mf = np.minimum(self.brightness_mfs[consequent_mf_name], rule_strength)

            # Store the clipped MF. If multiple rules activate the same consequent,
            # we take the maximum (union) of their clipped MFs.
            if consequent_mf_name not in activated_consequents:
                activated_consequents[consequent_mf_name] = clipped_mf
            else:
                activated_consequents[consequent_mf_name] = np.fmax(
                    activated_consequents[consequent_mf_name], clipped_mf
                )
        return activated_consequents

    def aggregate(self, activated_consequents):
        """
        Aggregates all activated consequent fuzzy sets into a single output fuzzy set.
        Args:
            activated_consequents (dict): {output_mf_name: clipped_mf_array}
        Returns:
            np.array: The aggregated output fuzzy set.
        """
        if not activated_consequents:
            return np.zeros_like(self.brightness_universe, dtype=float)  # No rules fired

        # Aggregate using the maximum (union) operator for Mamdani
        aggregated_output = np.zeros_like(self.brightness_universe, dtype=float)
        for mf_array in activated_consequents.values():
            aggregated_output = np.fmax(aggregated_output, mf_array)
        return aggregated_output

    def defuzzify_centroid(self, aggregated_output):
        """
        Defuzzifies the aggregated fuzzy set using the Centroid method.
        Args:
            aggregated_output (np.array): The aggregated output fuzzy set.
        Returns:
            float: The crisp defuzzified value.
        """
        # Centroid formula: sum(x * mu(x)) / sum(mu(x))
        # where x is the value in the universe and mu(x) is its membership degree.
        numerator = np.sum(self.brightness_universe * aggregated_output)
        denominator = np.sum(aggregated_output)

        if denominator == 0:
            return 0.0  # Avoid division by zero if no rules fired or aggregation is empty
        return numerator / denominator

    def compute(self, ambient_light, battery_level, time_of_day):
        """
        Runs the full fuzzy inference process.
        Returns:
            float: The crisp defuzzified brightness value in Nits.
        """
        inputs = {
            'ambient_light': ambient_light,
            'battery_level': battery_level,
            'time_of_day': time_of_day
        }

        # Step 1 & 2: Fuzzification and Inference
        activated_consequents = self.infer(inputs)

        # Step 3: Aggregation
        aggregated_output = self.aggregate(activated_consequents)

        # Step 4: Defuzzification
        crisp_output_nits = self.defuzzify_centroid(aggregated_output)
        return crisp_output_nits


# --- Function to draw membership functions ---
def draw_memberships():
    """
    Draws the membership functions used in the project.
    """
    controller = ManualFuzzyController()  # Create an instance to access universes and MFs

    # Plot Ambient Light MFs
    plt.figure(figsize=(8, 5))
    for name, mf_array in controller.ambient_light_mfs.items():
        plt.plot(controller.ambient_light_universe, mf_array, label=name)
    plt.title('Ambient Light Membership Functions')
    plt.xlabel('Ambient Light (0-100)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Battery Level MFs
    plt.figure(figsize=(8, 5))
    for name, mf_array in controller.battery_level_mfs.items():
        plt.plot(controller.battery_level_universe, mf_array, label=name)
    plt.title('Battery Level Membership Functions')
    plt.xlabel('Battery Level (0-100%)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Time of Day MFs
    plt.figure(figsize=(8, 5))
    for name, mf_array in controller.time_of_day_mfs.items():
        plt.plot(controller.time_of_day_universe, mf_array, label=name)
    plt.title('Time of Day Membership Functions')
    plt.xlabel('Time of Day (0-23 hours)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Brightness Output MFs
    plt.figure(figsize=(8, 5))
    for name, mf_array in controller.brightness_mfs.items():
        plt.plot(controller.brightness_universe, mf_array, label=name)
    plt.title('Brightness Output Membership Functions (Nits)')
    plt.xlabel('Brightness (Nits)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Tkinter GUI Application ---
class BrightnessApp:
    def __init__(self, master):
        self.master = master
        master.title("Fuzzy Brightness Recommender (Manual)")
        master.geometry("500x550")  # Increased height for new button

        # Initialize manual fuzzy controller
        self.fuzzy_controller = ManualFuzzyController()

        # Input variables and their ranges
        self.inputs = {
            'ambient_light': {'label': 'Ambient Light (0-100)', 'range': (0, 100), 'initial': 50},
            'battery_level': {'label': 'Battery Level (0-100%)', 'range': (0, 100), 'initial': 75},
            'time_of_day': {'label': 'Time of Day (0-23 hours)', 'range': (0, 23), 'initial': 12},
            'max_nits': {'label': 'Phone Max Nits (100-1500)', 'range': (100, 1500), 'initial': 800}
            # New input for max nits
        }

        self.create_widgets()
        self.update_brightness()  # Initial calculation

    def create_widgets(self):
        # Frame for sliders
        input_frame = ttk.LabelFrame(self.master, text="Input Conditions", padding="10 10")
        input_frame.pack(padx=10, pady=10, fill="x", expand=True)

        self.sliders = {}
        self.vars = {}
        self.formatted_vars = {}  # New dictionary to hold formatted string variables

        for i, (key, info) in enumerate(self.inputs.items()):
            # Label for the slider
            label = ttk.Label(input_frame, text=info['label'])
            label.grid(row=i, column=0, sticky="w", pady=5, padx=5)

            # Tkinter variable to hold slider value
            self.vars[key] = tk.DoubleVar(value=info['initial'])
            self.formatted_vars[key] = tk.StringVar()  # Create a StringVar for formatted display

            # Trace the DoubleVar to update the StringVar with formatted value
            self.vars[key].trace_add("write", lambda name, index, mode, k=key: self.update_formatted_value(k))

            # Horizontal slider (Scale widget)
            slider = ttk.Scale(
                input_frame,
                from_=info['range'][0],
                to=info['range'][1],
                orient="horizontal",
                variable=self.vars[key],
                command=lambda val, k=key: self.update_brightness()  # Pass key to ensure correct value is read
            )
            slider.grid(row=i, column=1, sticky="ew", pady=5, padx=5)
            self.sliders[key] = slider

            # Value display for the slider, linked to the new formatted_vars
            value_label = ttk.Label(input_frame, textvariable=self.formatted_vars[key])
            value_label.grid(row=i, column=2, sticky="w", pady=5, padx=5)

            # Initial update for formatted value
            self.update_formatted_value(key)

        # Configure column weights so the slider expands
        input_frame.grid_columnconfigure(1, weight=1)

        # Output display
        output_frame = ttk.LabelFrame(self.master, text="Recommended Brightness", padding="10 10")
        output_frame.pack(padx=10, pady=10, fill="x", expand=True)

        self.brightness_nits_label = ttk.Label(
            output_frame,
            text="Nits: --",
            font=("Helvetica", 18, "bold")
        )
        self.brightness_nits_label.pack(pady=5)

        self.brightness_percent_label = ttk.Label(
            output_frame,
            text="Percentage: --%",
            font=("Helvetica", 18, "bold")
        )
        self.brightness_percent_label.pack(pady=5)

        # Button to show membership functions
        self.show_mf_button = ttk.Button(self.master, text="Show Membership Functions", command=draw_memberships)
        self.show_mf_button.pack(pady=10)

    def update_formatted_value(self, key):
        """Updates the StringVar for a given key with the formatted value."""
        current_value = self.vars[key].get()
        # Format to two decimal places, ensuring at least one digit before decimal
        formatted_value = f"{current_value:.2f}"
        self.formatted_vars[key].set(formatted_value)

    def update_brightness(self):
        # Get current values from sliders
        ambient_light = self.vars['ambient_light'].get()
        battery_level = self.vars['battery_level'].get()
        time_of_day = self.vars['time_of_day'].get()
        max_nits = self.vars['max_nits'].get()  # Get max nits from the new slider

        # Compute recommended brightness in Nits using the manual fuzzy controller
        recommended_nits = self.fuzzy_controller.compute(
            ambient_light, battery_level, time_of_day
        )

        # Ensure recommended_nits does not exceed max_nits for the specific phone
        recommended_nits = min(recommended_nits, max_nits)
        recommended_nits = max(recommended_nits, 0)  # Ensure it's not negative

        # Calculate the percentage based on the phone's max nits
        percentage_brightness = (recommended_nits / max_nits) * 100 if max_nits > 0 else 0

        # Format for display
        formatted_nits = f"{recommended_nits:.1f}"
        formatted_percentage = f"{percentage_brightness:.1f}"

        # Update the brightness labels
        self.brightness_nits_label.config(text=f"Nits: {formatted_nits}")
        self.brightness_percent_label.config(text=f"Percentage: {formatted_percentage}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = BrightnessApp(root)
    # draw_memberships()
    root.mainloop()
