import numpy as np
import matplotlib.pyplot as plt

# Given values
F = 20 * 10**9  # 20 Gbits in bits
us = 15 * 10**6  # 15 Mbps in bits per second
d = 4 * 10**6  # 4 Mbps in bits per second
N = np.arange(1, 1001)  # Range of N values

# Calculate the minimum distribution time for client-server distribution
T_client_server = F / us

# Calculate the minimum distribution time for P2P distribution for each upload rate
upload_rates = [100 * 10**3, 600 * 10**3, 4 * 10**6]  # in bits per second

T_P2P = []
for u in upload_rates:
    T_p2p = np.maximum(F / np.minimum(us, d), F / (N * u + d))
    T_P2P.append(T_p2p)

# Plotting the curves
plt.figure(figsize=(10, 6))

# Client-Server Distribution curve
plt.plot(N, np.full_like(N, T_client_server), label='Client-Server Distribution', color='blue')

# P2P Distribution curves for different upload rates
labels = ['P2P Distribution with u = 100 Kbps', 'P2P Distribution with u = 600 Kbps', 'P2P Distribution with u = 4 Mbps']
colors = ['orange', 'green', 'red']

for i in range(len(upload_rates)):
    plt.plot(N, T_P2P[i], label=labels[i], color=colors[i], linestyle='dashed')

plt.title('Comparison of Minimum Distribution Time in Client-Server and P2P Distribution')
plt.xlabel('Number of Peers (N)')
plt.ylabel('Minimum Distribution Time (seconds)')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('plot.png')
