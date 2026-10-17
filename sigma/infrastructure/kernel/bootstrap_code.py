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

# The R kernel bridges to the Python ``sigma`` helper through reticulate, so an R notebook can load and
# save Sigma Algebra datasets as data.frames. ``RETICULATE_PYTHON`` points reticulate at the server's venv.
R_BOOTSTRAP_CODE = """
library(reticulate)
sigma <- reticulate::import("sigma.infrastructure.kernel.helpers")$sigma
"""
