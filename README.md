# Bachelor Thesis - Simulating Body Growth in a Human Infant Model

This repository contains additional code that is not directly connected to the growth of MIMo and thus can't be found in the [main repository](https://github.com/trieschlab/MIMo).

The main contributions of this repository are:
- `plot.py` : Creates most of the plots for the thesis.
- `show.py` : Loads MIMo in the MuJoCo viewer and lets him grow or performs a strength test.
- `media/` : Figures from the thesis and additional material such as the growth video.



## Installation

- `git clone https://github.com/miles-lenz/thesis-bachelor.git`
-  `pip install -r requirements.txt`
    
## Usage

The two main scripts can be called with the following command:
```python
python <file> <function_name> <kwargs>
```
Explanation
- `<file>` needs to be `plot.py` or `show.py`.
- `<function_name>` needs to be the exact name of the function within the script.
- `<kwargs>` are one ore more arguments in the form of `key=value` that match the function signature.



## Acknowledgements

- All growth charts within `resources/growth_charts/` were downloaded from the [official WHO website](https://www.who.int/tools/child-growth-standards/standards).
- Files within `resources/mimoEnv/` and `resources/mimoGrowth` are copied from the [main repository](https://github.com/trieschlab/MIMo).
