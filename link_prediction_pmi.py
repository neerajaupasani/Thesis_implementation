# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['-3.0','-2.0', '-1.0', '0.0', '1.0', '2.0', '3.0']


y_values=[ 0.7, 0.8, 0.9, 1.0]
my_yticks=[ '0.7', '0.8', '0.9', '1.0']

pmi=[ 0.905, 0.960, 0.974, 0.978, 0.980, 0.981, 0.988]

ax1.plot(x_values, pmi,'r-^',lw=3.5, markersize=12, label='pmi')


ax1.set_title('Cora' , fontsize=25)
ax1.set_ylabel('AUC', fontsize=25)
ax1.set_xlabel(r'$\alpha $', fontsize=25)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=25)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=25)
ax1.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()

pmi=[ 0.895, 0.933, 0.947, 0.943, 0.950, 0.951, 0.949 ]

ax2.plot(x_values, pmi,'r-^',lw=3.5, markersize=12, label='pmi')


ax2.set_title('Citeseer' , fontsize=25)
ax2.set_ylabel('AUC', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks, fontsize=25)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

pmi=[ 0.710, 0.839, 0.942, 0.963, 0.976, 0.986, 0.988]

ax3.plot(x_values, pmi,'r-^',lw=3.5, markersize=12, label='pmi')


ax3.set_title('Wiki' , fontsize=25)
ax3.set_ylabel('AUC', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)

ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks, fontsize=25)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
ax3.tick_params(axis='both',labelsize=25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend( handles, labels, loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=1, fontsize=25 )

plt.savefig('link_prediction_pmi.pdf',  bbox_inches='tight',dpi=100)



plt.show()
