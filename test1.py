import matplotlib.pyplot as plt


ax = plt.gca()
ax.set_xlim(0, 200000)
miloc = plt.MultipleLocator(10000)
ax.xaxis.set_minor_locator(miloc)
ay = plt.gca()
ay.set_ylim(0, 1)
miloc = plt.MultipleLocator(60)
ay.yaxis.set_minor_locator(miloc)
ax.grid(axis='y', which='minor')

plt.show()
# plt.grid(b=True, which='minor', axis='y', linestyle='--')
# # (b, which, axis, color, linestyle, linewidth， **kwargs)
# plt.show()