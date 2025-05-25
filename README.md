# Fuzzy Screen Brightness Controller

An intelligent screen brightness adjustment system using fuzzy logic principles.

## Overview

This project implements a fuzzy logic controller to automatically adjust screen brightness based on ambient light conditions and user preferences. By using fuzzy logic instead of crisp thresholds, the system provides smoother, more human-like transitions between brightness levels.

## Features

- Real-time ambient light detection
- User preference integration
- Smooth brightness transitions
- Energy-efficient operation
- Customizable rule base

## Implementation

The system uses the following fuzzy input variables:

- Ambient light level
- Screen usage time
- Time of the day
- Battery status

Output:

- Screen brightness level

## Requirements

- Python 3.7+
- NumPy
- scikit-fuzzy
- Matplotlib (for visualization)

## Usage

```bash
python fuzzy_brightness.py
```

## Project Structure

```
fuzzy_screen_brightness/
├── main.py    # Main implementation
```
