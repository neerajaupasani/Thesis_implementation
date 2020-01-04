# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['0','0.01', '0.1', '0.2', '0.3', '0.5', '0.8']


y_values=[ 0.5,0.6,0.7, 0.8, 0.9, 1.0]
my_yticks=[ '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

cos=[0.965,0.961,0.936,0.879,0.765,0.502,0.501]
jac=[0.955,0.949,0.857,0.667,0.555,0.510,0.516 ]
hpi=[0.959,0.957,0.958,0.945,0.924,0.652,0.629 ]
hdi=[ 0.943,0.938,0.878,0.740,0.632,0.522,0.513]

ax1.plot(x_values, cos,'m->',lw=3.5, markersize=12, label='Cosine')
ax1.plot(x_values, jac,'r-o',lw=3.5, markersize=12, label='Jaccard')
ax1.plot(x_values, hpi,'b-d',lw=3.5, markersize=12,label='HPI' )
ax1.plot(x_values, hdi,'g-s',lw=3.5, markersize=12, label='HDI')
#ax1.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='6' )
#ax1.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax1.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='8')
#ax1.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')
#ax1.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')

ax1.set_title('Cora' , fontsize=25)
ax1.set_ylabel('AUC score', fontsize=25)
ax1.set_xlabel(r'$\alpha $', fontsize=25)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=25)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=25)
ax1.tick_params(axis='both',labelsize=25)
#ax1.set_yticks()
#ax1.tick_params(axis='both',labelsize=20)
#ax1.yaxis.tick_right()

cos=[ 0.940,0.937,0.935,0.905,0.859,0.557,0.508]
jac=[0.943,0.937,0.900,0.708,0.565,0.512,0.506 ]
hpi=[0.934,0.936,0.935,0.930,0.923,0.802,0.731 ]
hdi=[0.942,0.945,0.926,0.781,0.670,0.532,0.505 ]

ax2.plot(x_values, cos,'m->',lw=3.5, markersize=12, label='Cosine')
ax2.plot(x_values, jac,'r-o',lw=3.5, markersize=12, label='Jaccard')
ax2.plot(x_values, hpi,'b-d',lw=3.5, markersize=12,label='HPI' )
ax2.plot(x_values, hdi,'g-s',lw=3.5, markersize=12, label='HDI')
#ax2.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='6' )
#ax2.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax2.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='8')
#ax2.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')
#ax2.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')




ax2.set_title('Citeseer' , fontsize=25)
ax2.set_ylabel('AUC score', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()


cos=[0.951,0.952,0.934,0.877,0.822,0.678,0.568 ]
jac=[0.953,0.947,0.863,0.770,0.683,0.568,0.552 ]
hpi=[0.956,0.954,0.951,0.924,0.917,0.867,0.801 ]
hdi=[ 0.944,0.945,0.885,0.805,0.757,0.610,0.559]

ax3.plot(x_values, cos,'m->',lw=3.5, markersize=12, label='Cosine')
ax3.plot(x_values, jac,'r-o',lw=3.5, markersize=12, label='Jaccard')
ax3.plot(x_values, hpi,'b-d',lw=3.5, markersize=12,label='HPI' )
ax3.plot(x_values, hdi,'g-s',lw=3.5, markersize=12, label='HDI')
#ax3.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='6' )
#ax3.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax3.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='8')
#ax3.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')
#ax3.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')


ax3.set_title('Wiki' , fontsize=25)
ax3.set_ylabel('AUC score', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)
ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
#ax1.set_yticks()
ax3.tick_params(axis='both',labelsize=25)

handles, labels = ax1.get_legend_handles_labels()
legend=fig.legend( handles, labels, loc='upper center' , bbox_to_anchor=(0.42, 1.25) ,ncol=5, fontsize=20 )

 #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')

#plt.title('Performance of methods in event-to-user recommendation')
plt.setp(legend.get_title(),fontsize=25)

plt.savefig('link_prediction.pdf',  bbox_inches='tight',dpi=100)



plt.show()
