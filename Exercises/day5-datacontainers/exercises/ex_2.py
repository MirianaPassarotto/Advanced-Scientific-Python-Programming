import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#### a. Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF),
#  cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable
mu = 0.6
mean, var, skew, kurt = stats.poisson.stats(mu, moments='mvsk')
x = np.arange(stats.poisson.ppf(0.01, mu), stats.poisson.ppf(0.99, mu))

# (1) Plot Probability Mass Function (PMF)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.stem(x, stats.poisson.pmf(x, mu))
ax.set_title("Poisson Distribution PMF (μ = 0.6)")
ax.set_xlabel("Number of occurrences")
ax.set_ylabel("Probability")
plt.grid()
plt.show()

# (2) Plot Cumulative Distribution Function (CDF)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(x, stats.poisson.cdf(x, mu))
ax.set_title("Poisson Distribution CDF (μ = 0.6)")
ax.set_xlabel("Number of occurrences")
ax.set_ylabel("Cumulative Probability")
plt.grid()
plt.show()

# (3) Generate 1000 random realizations and plot histogram
random_samples = stats.poisson.rvs(mu, size=1000)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.hist(random_samples, bins=np.arange(0, np.max(random_samples)+2)-0.5, density=True, alpha=0.7, edgecolor='black')
ax.set_title("Histogram of 1000 Poisson Random Samples (μ = 0.6)")
ax.set_xlabel("Number of occurrences")
ax.set_ylabel("Frequency")
plt.grid()
plt.show()

#### b. Create a continious random variable with normal distribution and plot its probability mass function (PMF),
#  cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu = 0       
sigma = 1    

# create x values for plot
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000) 

# (1) Plot Probability Density Function (PDF)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(x, stats.norm.pdf(x, mu, sigma),label="PDF")
ax.fill_between(x, stats.norm.pdf(x, mu, sigma))
ax.set_title("Normal Distribution PDF (mu = 0, sigma = 1)")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.grid()
plt.show()

# (2) Plot Cumulative Distribution Function (CDF)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(x, stats.norm.cdf(x, mu, sigma),label="CDF")
ax.set_title("Normal Distribution CDF (mu = 0, sigma = 1)")
ax.set_xlabel("Value")
ax.set_ylabel("Cumulative Probability")
ax.legend()
plt.grid()
plt.show()

# (3) Generate 1000 random realizations and plot histogram
random_samples = stats.norm.rvs(mu, sigma, size=1000)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.hist(random_samples, bins=30, density=True, label="Histogram")
ax.plot(x, stats.norm.pdf(x, mu, sigma))  # Overlay theoretical PDF
ax.set_title("Histogram of 1000 Normal Random Samples (mu = 0, sigma = 1)")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.grid()
plt.show()



#### c. Test if two sets of (independent) random data comes from the same distribution
#Hint: Have a look at the ```ttest_ind``` 

data1 = np.random.normal(loc=5, scale=2, size=1000) 
data2 = np.random.normal(loc=5, scale=2, size=1000)  

t_stat, p_value = stats.ttest_ind(data1, data2)

print(f"T-statistic: {t_stat:}")
print(f"P-value: {p_value:}")

#
alpha = 0.05  # Significance level
if p_value < alpha:
    print("datasets are from different distributions")
else:
    print(" datasets may be from the same distribution")
