from functions import*
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

small = False
reference_lines = False
fig, axs = plt.subplots(figsize=(9, 6), nrows=2, ncols=1)
# elements = np.array([487, 983, 1484, 1981, 2984, 3983])
# memory = np.loadtxt("results/Q_benchmark_memory", delimiter=",")
# time = np.loadtxt("results/Q_benchmark_time", delimiter=",")
# elements = np.loadtxt("results/Q_benchmark_elements", delimiter=",")
memory = np.loadtxt("Q_benchmark_memory_with_gpu_1", delimiter=",")
time = np.loadtxt("Q_benchmark_time_with_gpu_1", delimiter=",")
elements = np.loadtxt("Q_benchmark_elements_with_gpu_1", delimiter=",")
memory_large = np.loadtxt("Q_benchmark_large_samples_memory", delimiter=",")
time_large = np.loadtxt("Q_benchmark_large_samples_time", delimiter=",")
time_large[0, 0] = time_large[0, 0]/5
elements_large = np.loadtxt("Q_benchmark_large_samples_elements", delimiter=",")
# fig hotfix
# elements_large = np.sort(np.concatenate((elements_large, np.array([40000]))))
# time_large = np.vstack((np.sort(np.concatenate((time_large[0], np.array([239.4])))),
#                         np.sort(np.concatenate((time_large[1], np.array([147]))))))[:, 0:]
# memory_large = np.vstack((np.sort(np.concatenate((memory_large[0], np.array([4.944])))),
#                           np.sort(np.concatenate((memory_large[1], np.array([2.7465]))))))[:, 0:]
# for now fix the data for float16 gpu version
time[0, 0] = 1.1042
# for now fix the data for elements of large elements
elements_large[-1] = 200000
elements_large[-2] = 100000
elements_large[-3] = 40000

width = 250
widths = 1/3.5*np.array([5000, 10000, 20000, 50000, 100000])
axs[0].set_ylabel("Peak memory usage in MB", fontsize=14)
axs[1].set_ylabel("Time", fontsize=14)
y_ticks = [1, 10, 60, 600, 3600, 5*3600]
y_labels = ["1s", "10s", "1 min", "10 min", "1h", "5h"]
length = elements.shape[0]
displacements = [200, 200, 300, 200, 200]
color = ['green', 'orange', 'red', 'purple']
if small:
        axs[0].set_ylim([1e-2, 1e4])
        axs[1].set_ylim([1e-1, 1e2])
        for i in range(4):
                axs[0].bar(elements - 500 + ((50 + width)*i), memory[i], width=width, edgecolor='black', color=color[i])
                axs[1].bar(elements - 500 + ((50 + width)*i), time[i], width=width, edgecolor='black', color=color[i])

else:
        axs[0].set_ylim([1e-1, 1e2])
        axs[1].set_ylim([1, 2e4])

        axs[0].bar(elements_large - 0.4*widths, memory_large[0], width=0.8*widths, edgecolor='black', color='red')
        axs[1].bar(elements_large - 0.4*widths, time_large[0], width=0.8*widths, edgecolor='black', color='red')
        axs[0].bar(elements_large + 0.6*widths, memory_large[1], width=widths, edgecolor='black', color='purple')
        axs[1].bar(elements_large + 0.6*widths, time_large[1], width=widths, edgecolor='black', color='purple')
        # axs[0].plot(elements_large[:5], memory_large[0][:5], "x--", alpha=0.9, lw=2, color='red')
        # axs[1].plot(elements_large[:5], time_large[0][:5], "x--", alpha=0.9, lw=2, color='red')
        # axs[0].plot(elements_large[:5], memory_large[1][:5], "x--", alpha=0.9, lw=2, color='purple')
        # axs[1].plot(elements_large[:5], time_large[1][:5], "x--", alpha=0.9, lw=2, color='purple')
if small:
        for j in range(2):
                axs[j].set_yscale("log")
                axs[j].set_axisbelow(True)
                axs[j].grid(True, which='both', axis='y', zorder=-25.0)
                axs[j].set_xlabel("Number of elements", fontsize=14)
                # axs[j].legend(["python loop", "numba", "jacobi", "vectorized jacobi", "vectorized jacobi (cupy)"],
                #               loc="upper left", fontsize=9)
                # axs[j].legend(["Python", "numba", "Jacobi", "v-Jacobi", "v-Jacobi (cupy)"],
                #               loc="upper center", fontsize=10)
                for label in axs[j].get_xticklabels():
                        label.set_fontsize(14)
                axs[j].set_xlim([1000, 11000])
        if reference_lines:
                axs[0].axhline(y=1, color='k', alpha=0.3)
                axs[0].axhline(y=1e1, color='k', alpha=0.3)
                axs[0].axhline(y=1e2, color='k', alpha=0.3)
                axs[0].axhline(y=1e3, color='k', alpha=0.3)
        for e in y_ticks[:-1]:
                axs[1].axhline(y=e, color='k', alpha=0.5, zorder=-25.0)
        axs[1].set_yticks(y_ticks[:-3])
        for label in axs[0].get_yticklabels():
                label.set_fontsize(14)
        axs[1].set_yticklabels(y_labels[:-3], fontsize=14)
        # plt.close()
        # fig1, ax1 = plt.subplots()
        # a = np.random.rand(5, 10)
        # for i in range(5):
        #         ax1.plot(a[i, :], a[i, :])
        # ax1.legend(["Python", "numba", "Jacobi", "v-Jacobi", "v-Jacobi (cupy)"],
        #                       loc="best", fontsize=10)
        # ax1.axis(False)
        # label_params = ax1.get_legend_handles_labels()
        # fig2, ax2 = plt.subplots()
        # ax2.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":50})
        # ax2.axis(False)
        # # fig2.savefig("q_small_label.pdf", format='pdf', dpi=600)
        # plt.show()

else:
        for j in range(2):
                axs[j].set_xscale("log")
                axs[j].set_yscale("log")
                axs[j].set_axisbelow(True)
                axs[j].grid(True, which='both', axis='y', zorder=-25.0)
                axs[j].set_xticks([x for x in elements_large[0:6]])
                axs[j].set_xticklabels([str(np.round(x, -3))[:-2] for x in elements_large[0:6]], fontsize=14)
                axs[j].set_xlim(8300, 250000)
                axs[j].set_xlabel("Number of elements", fontsize=14)
                # axs[j].legend(["vectorized jacobi", "vectorized jacobi (cupy)"],
                #               loc="upper left", fontsize=10)
                for label in axs[j].get_xticklabels():
                        label.set_fontsize(12)
        if reference_lines:
                axs[0].axhline(y=1, color='k', alpha=0.3)
                axs[0].axhline(y=1e1, color='k', alpha=0.3)
                axs[0].axhline(y=1e2, color='k', alpha=0.3)
                axs[0].axhline(y=1e3, color='k', alpha=0.3)
                axs[0].axhline(y=1e4, color='k', alpha=0.3)

        for e in y_ticks:
                axs[1].axhline(y=e, color='k', alpha=0.5, zorder=-25.0)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels, fontsize=12)
        # for label in axs[0].get_yticklabels():
        #         label.set_fontsize(12)


fig.tight_layout()
if small:
        plt.savefig("small_q_methods.svg", format='svg', dpi=600)
else:
        plt.savefig("large_q_methods.svg", format='svg', dpi=600)
plt.show()
