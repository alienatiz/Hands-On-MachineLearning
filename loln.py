# Law of large number
import numpy as np
from scipy.stats import binom

from settings import *

# Use binom.cdf from scipy.stats
# Estimate the probability which the times that a coin is tossed
print(1 - binom.cdf(49, 100, 0.51))
print(1 - binom.cdf(499, 1000, 0.51))
print(1 - binom.cdf(4999, 10000, 0.51))

head_proba = 0.51
coin_tossed = (np.random.rand(10000, 10) < head_proba).astype(np.int32)
total_ratio = np.cumsum(coin_tossed, axis=0) / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(total_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.51, 0.51], "k-", label="50%")
plt.xlabel("The number of times that a coin is tossed")
plt.ylabel("The number of heads")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
save_fig("loln_graph_plot")
plt.show()
