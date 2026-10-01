"""Code run silently when a console kernel starts, so the usual libraries and ``sigma`` are ready."""

BOOTSTRAP_CODE = """
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sigma.infrastructure.kernel.helpers import sigma

pd.options.display.max_columns = 50
pd.options.display.width = 120
plt.rcParams["figure.figsize"] = (7.5, 4.2)
plt.rcParams["figure.dpi"] = 100
"""
