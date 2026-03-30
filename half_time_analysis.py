import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load normalized data
data_path = '/Users/nataliaionescu/Desktop/AB42_project/fits/pH_6.5/6.5_without_1.4.tsv'
df = pd.read_csv(data_path, sep='\t', header=1)
current_df = df
x_actual_data = []
y_actual_data = []

for i in range(len(current_df.columns) //2):
    x_current = current_df.iloc[:, i*2].values
    y_current = current_df.iloc[:, i*2+1].values
    x_actual_data.append(x_current[~np.isnan(x_current)])
    y_actual_data.append(y_current[~np.isnan(y_current)])

m0vals = [0.8, 0.8, 0.8, 1.1, 1.1, 1.1, 1.4, 1.4, 1.9, 1.9, 1.9, 2.5, 2.5, 2.5, 3.4, 3.4, 3.4, 4.5, 4.5, 4.5, 6.0, 6.0, 6.0]

# extract half times from normalized data
def get_half_time(time, signal):
    for i in range(len(signal) -1):
        if signal[i] <= 0.5 <= signal[i+1]:
        # linearly interpoalate
            slope = (0.5 - signal[i]) / (time[i+1] - time[i])
            half_time = time[i] + slope * (time[i+1] - time[i])
    return half_time

# get half time for each m0
half_times = []
for i in range(len(x_actual_data)):
    t_half = get_half_time(x_actual_data[i], y_actual_data[i])
    half_times.append(t_half)

# turn into np array to mask for the same m0 values within triplicates
half_times = np.array(half_times)
m0vals = np.array(m0vals)
unique_m0 = sorted(set(m0vals))
mean_half_times = []
for m0 in unique_m0:
    mask = m0vals == m0
    mean_half_times.append(half_times[mask].mean())

unique_m0 = np.array(unique_m0)
# log-log plot of t_half vs m0
plt.figure(figsize=(7,7))
plt.scatter(np.log10(unique_m0), np.log10(mean_half_times))
plt.show()

# separate low and high m0 values
n = len(unique_m0)
mid = n // 2
lower_m0 = unique_m0[:mid]
higher_m0 = unique_m0[mid:]
lower_t = mean_half_times[:mid]
higher_t = mean_half_times[mid:]

# fit different lines to the two regimes of m0 values
lower_slope, lower_intercept, lower_r, lower_p, lower_se = stats.linregress(np.log10(lower_m0), np.log10(lower_t))
higher_slope, higher_intercept, higher_r, higher_p, higher_se = stats.linregress(np.log10(higher_m0), np.log10(higher_t))

print('low m0 slope', lower_slope)
print('high m0 slope', higher_slope)

# add fitted lines to half-time plot
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(np.log10(lower_m0),np.log10(lower_t), color='steelblue', label='low m0')
ax.scatter(np.log10(higher_m0),np.log10(higher_t), color='tomato', label='high m0')
ax.set_xlabel('log m0')
ax.set_ylabel('log t_half')

def draw_line(m0, slope, intercept, color, label, se):
    x = np.linspace(np.log10(m0) - 0.3, np.log10(m0) + 0.3, 100)
    ax.plot(x, slope*x + intercept, linestyle='--', color=color, label=f'{label}: slope = {slope:.2f} +- {se:.2f}')

draw_line(lower_m0[2], lower_slope, lower_intercept, 'steelblue', 'lower m0', lower_se)
draw_line(higher_m0[2], higher_slope, higher_intercept, 'tomato', 'higher m0', higher_se)

plt.legend()
plt.show()



