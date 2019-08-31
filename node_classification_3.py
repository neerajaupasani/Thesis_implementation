# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['0.0','0.01', '0.1', '0.2', '0.3', '0.5', '0.8']


y_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
my_yticks=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']

cos=[ 0.71009,0.71009,0.70605,0.68185,0.58201,0.36902,0.17933 ]
jac=[0.67451,0.67451,0.65536,0.44286,0.34037,0.18339,0.14328 ]
lhn1=[0.49709,0.49709,0.38472,0.25834,0.21028,0.09715,0.09715 ]
sor=[0.70766,0.70766,0.69581,0.64525,0.55004,0.30836,0.15495 ]
hpi=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265 ]
hdi=[ 0.67465,0.67465,0.67011,0.56654,0.48307,0.22532,0.14328 ]



ax1.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax1.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax1.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax1.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax1.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax1.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')

ax1.set_title('Cora', fontsize=22)
ax1.set_ylabel('F-Macro', fontsize=22)
ax1.set_xlabel(r'$\alpha $', fontsize=22)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=22)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=22)
ax1.tick_params(axis='both',labelsize=22)
#ax1.set_yticks()
#ax1.tick_params(axis='both',labelsize=20)
#ax1.yaxis.tick_right()

cos=[ 0.46006,0.46006,0.45612,0.45774,0.44387,0.28147,0.19141]
jac=[ 0.43670,0.43853,0.42812,0.34355,0.29405,0.18347,0.15791]
lhn1=[ 0.34835,0.34748,0.30463,0.26349,0.22263,0.15921,0.15921]
sor=[ 0.46768,0.46768,0.46617,0.46207,0.41427,0.26245,0.15993]
hpi=[ 0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.35082]
hdi=[ 0.46776,0.46776,0.46658,0.42080,0.36968,0.20371,0.16607]

ax2.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax2.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax2.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax2.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax2.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax2.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')

ax2.set_title('Citeseer', fontsize=25)
ax2.set_ylabel('F-Macro', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

cos=[ 0.53969,0.53969,0.54204,0.51450,0.47668,0.39781,0.13228  ]
jac=[ 0.52770,0.52770,0.50556,0.45553,0.39779,0.22683,0.09120  ]
lhn1=[ 0.28255,0.27771,0.14375,0.10847,0.08921,0.06971,0.06971  ]
sor=[ 0.53253,0.53313,0.53739,0.50516,0.47666,0.36938,0.11377  ]
hpi=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327  ]
hdi=[ 0.53678,0.53661,0.51976,0.48635,0.44604,0.30493,0.10712  ]


ax3.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax3.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax3.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax3.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax3.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax3.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')

ax3.set_title('Wiki', fontsize=25)
ax3.set_ylabel('F-Macro', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)
ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
#ax1.set_yticks()
ax3.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()




handles, labels = ax1.get_legend_handles_labels()
fig.legend( handles, labels, loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=6, fontsize=22 ) #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')

#plt.title('Performance of methods in event-to-user recommendation')
plt.savefig('node_classification.pdf',  bbox_inches='tight',dpi=100)



plt.show()
