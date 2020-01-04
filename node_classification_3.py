# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['0','0.01', '0.1', '0.2', '0.3', '0.5', '0.8']


y_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
my_yticks=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']

cos=[0.71009,0.71009,0.70605,0.68185,0.58201,0.17632,0.17632 ]
jac=[0.67451,0.67451,0.65536,0.44286,0.34037,0.14328,0.14328 ]
hpi=[0.74193,0.74193,0.73475,0.72572,0.70238,0.47265,0.47265 ]
hdi=[ 0.67465,0.67465,0.67011,0.56654,0.48307,0.14328,0.14328]


ax1.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
#ax1.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax1.plot(x_values, jac,'g-^',lw=3.5, markersize=12, label='Jaccard')
#ax1.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax1.plot(x_values, hpi,'c-o',lw=3.5, markersize=12, label='HPI')
#ax1.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax1.plot(x_values, hdi,'b->',lw=3.5, markersize=12, label='HDI')
#ax1.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')
#ax1.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='10')

ax1.set_title('Cora', fontsize=22)
ax1.set_ylabel('Macro-F1 score', fontsize=22)
ax1.set_xlabel(r'$\alpha $', fontsize=22)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=22)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=22)
ax1.tick_params(axis='both',labelsize=22)
#ax1.set_yticks()
#ax1.tick_params(axis='both',labelsize=20)
#ax1.yaxis.tick_right()

cos=[0.46006,0.46006,0.45612,0.45774,0.44387,0.28466,0.19453 ]
jac=[0.43905,0.43853,0.42812,0.34355,0.29405,0.18133,0.15615 ]
hpi=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031 ]
hdi=[0.46732,0.46342,0.46092,0.41999,0.36968,0.21249,0.17807 ]


ax2.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
#ax2.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax2.plot(x_values, jac,'g-^',lw=3.5, markersize=12, label='Jaccard')
#ax2.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax2.plot(x_values, hpi,'c-o',lw=3.5, markersize=12, label='HPI')
#ax2.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax2.plot(x_values, hdi,'b->',lw=3.5, markersize=12, label='HDI')
#ax2.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')
#ax2.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='10')

ax2.set_title('Citeseer', fontsize=25)
ax2.set_ylabel('Macro-F1 score', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)


cos=[0.53969,0.53969,0.54204,0.51450,0.47668,0.39781,0.13235 ]
jac=[0.52770,0.52770,0.50556,0.45553,0.39779,0.22683,0.09144 ]
hpi=[0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327 ]
hdi=[0.53678,0.53661,0.51976,0.48635,0.44604,0.30339,0.10778 ]

ax3.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
#ax3.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30')
ax3.plot(x_values, jac,'g-^',lw=3.5, markersize=12, label='Jaccard')
#ax3.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax3.plot(x_values, hpi,'c-o',lw=3.5, markersize=12, label='HPI')
#ax3.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax3.plot(x_values, hdi,'b->',lw=3.5, markersize=12, label='HDI')
#ax3.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')
#ax3.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='10')

ax3.set_title('Wiki', fontsize=25)
ax3.set_ylabel('Macro-F1 score', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)
ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
#ax1.set_yticks()
ax3.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()




handles, labels = ax1.get_legend_handles_labels()
legend=fig.legend( handles, labels, loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=5, fontsize=25 ) #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')


#plt.title('Performance of methods in event-to-user recommendation')
plt.savefig('node_classification.pdf',  bbox_inches='tight',dpi=100)



plt.show()
