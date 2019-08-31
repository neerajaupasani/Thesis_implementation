# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['0.0','0.01', '0.1', '0.2', '0.3', '0.5', '0.8']


y_values=[ 0.5,0.6,0.7, 0.8, 0.9, 1.0]
my_yticks=[ '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

cos=[ 0.950, 0.954, 0.937, 0.878, 0.775, 0.555, 0.502]
jac=[ 0.946, 0.952, 0.831, 0.653, 0.557, 0.503, 0.504]
lhn1=[ 0.888, 0.875, 0.602, 0.507, 0.502, 0.499, 0.496]
sor=[ 0.951, 0.944, 0.930, 0.838, 0.725, 0.537, 0.499]
hpi=[ 0.959, 0.959, 0.958,0.942,0.922,0.755,0.626]
hdi=[ 0.932, 0.939, 0.881,0.741,0.633,0.514,0.514]

ax1.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax1.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax1.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax1.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax1.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax1.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')

ax1.set_title('Cora' , fontsize=25)
ax1.set_ylabel('AUC', fontsize=25)
ax1.set_xlabel(r'$\alpha $', fontsize=25)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=25)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=25)
ax1.tick_params(axis='both',labelsize=25)
#ax1.set_yticks()
#ax1.tick_params(axis='both',labelsize=20)
#ax1.yaxis.tick_right()



cos=[ 0.935,0.941,0.935,0.898,0.855,0.562,0.507 ]
jac=[ 0.932,0.938,0.895,0.710,0.565,0.513,0.506  ]
lhn1=[ 0.895,0.899,0.609,0.510,0.507,0.502,0.501 ]
sor=[ 0.943,0.948,0.927,0.893,0.810,0.541,0.504 ]
hpi=[ 0.936,0.934,0.931,0.937,0.918,0.799,0.708 ]
hdi=[ 0.943,0.943,0.919,0.810,0.674,0.534,0.501 ]


ax2.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax2.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax2.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax2.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax2.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax2.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')


ax2.set_title('Citeseer' , fontsize=25)
ax2.set_ylabel('AUC', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()

cos=[ 0.954,0.953,0.934,0.875,0.820,0.678,0.561 ]
jac=[ 0.952,0.950,0.859,0.776,0.683,0.570,0.549 ]
lhn1=[ 0.896,0.851,0.538,0.508,0.507,0.504,0.503 ]
sor=[ 0.948,0.951,0.929,0.856,0.799,0.652,0.568 ]
hpi=[ 0.956,0.955,0.945,0.928,0.918,0.859,0.806 ]
hdi=[ 0.944,0.944,0.881,0.797,0.748,0.611,0.546 ]


ax3.plot(x_values, cos,'r-o',lw=3.5, markersize=12, label='Cosine')
ax3.plot(x_values, jac,'b-d',lw=3.5, markersize=12,label='Jaccard' )
ax3.plot(x_values, lhn1,'g-^',lw=3.5, markersize=12, label='LHN1')
ax3.plot(x_values, sor,'y-s',lw=3.5, markersize=12,label='Sorensen' )
ax3.plot(x_values, hpi,'m->',lw=3.5, markersize=12, label='HPI')
ax3.plot(x_values, hdi,'c-o',lw=3.5, markersize=12, label='HDI')

ax3.set_title('Wiki' , fontsize=25)
ax3.set_ylabel('AUC', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)
ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
#ax1.set_yticks()
ax3.tick_params(axis='both',labelsize=25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend( handles, labels, loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=6, fontsize=25 )

 #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')

#plt.title('Performance of methods in event-to-user recommendation')
plt.savefig('link_prediction.pdf',  bbox_inches='tight',dpi=100)



plt.show()
