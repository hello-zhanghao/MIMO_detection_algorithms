import numpy as np
import matplotlib.pyplot as plt

# results = np.load('ss', allow_pickle=True).item()
# for key, value in results.items():
#     if key == 'SNR_dBs':
#         pass
#     else:
#         print(key,value)
#         plt.plot(results['SNR_dBs'], value, label=key)
# plt.grid(True, which='minor', linestyle='--')
# plt.yscale('log')
# plt.xlabel('SNR')
# plt.ylabel('SER')
# plt.title('Nr%dNt%d_mod%s'%(params['Nr'], params['Nt'], params['modulation']))
# plt.legend()
# plt.show()

results = np.load('loss_all_DEtNetSIC.npy', allow_pickle=True).item()
for key, value in results.items():
   print(key, value)
   plt.plot(value, label=key)
plt.grid(True, which='minor', linestyle='--')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('SER')
# plt.title('Nr%dNt%d_mod%s'%(params['Nr'], params['Nt'], params['modulation']))
plt.legend()
plt.show()
