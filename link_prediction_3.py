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

tw=[0.941,0.951,0.883,0.745,0.635,0.513,0.512 ]
th=[0.944,0.938,0.876,0.758,0.626,0.511,0.524 ]
fo=[ 0.949,0.944,0.883,0.747,0.622,0.513,0.521]
#fi=[ 0.943,0.949,0.882,0.749,0.637,0.510,0.504]
#si=[0.947,0.953,0.881,0.739,0.630,0.514,0.516 ]
#se=[0.950,0.939,0.877,0.746,0.634,0.513,0.527 ]
#ei=[0.950,0.941,0.883,0.741,0.636,0.513,0.515 ]
#ni=[ 0.944,0.943,0.867,0.746,0.631,0.511,0.508]
te=[0.943,0.938,0.878,0.740,0.632,0.522,0.513 ]

ax1.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax1.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax1.plot(x_values, fo,'m->',lw=3.5, markersize=12, label='10')
ax1.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')
#ax1.plot(x_values, fi,'r->',lw=3.5, markersize=12, label='6')
#ax1.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='7' )
#ax1.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax1.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='9')
#ax1.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')

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

tw=[0.936,0.944,0.925,0.803,0.673,0.535,0.503 ]
th=[0.942,0.940,0.917,0.791,0.670,0.533,0.501 ]
fo=[ 0.944,0.938,0.924,0.805,0.650,0.536,0.505]
#fi=[ 0.941,0.941,0.921,0.803,0.661,0.537,0.505 ]
#si=[ 0.946,0.940,0.920,0.794,0.661,0.533,0.503]
#se=[0.931,0.944,0.914,0.803,0.671,0.533,0.504 ]
#ei=[0.941,0.938,0.919,0.802,0.658,0.533,0.503 ]
#ni=[0.941,0.947,0.919,0.805,0.658,0.537,0.503 ]
te=[0.942,0.945,0.926,0.781,0.670,0.532,0.505  ]

ax2.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax2.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax2.plot(x_values, fo,'m->',lw=3.5, markersize=12, label='10')
ax2.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')
#ax2.plot(x_values, fi,'r->',lw=3.5, markersize=12, label='6')
#ax2.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='7' )
#ax2.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax2.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='9')
#ax2.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')





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


tw=[0.945,0.944,0.886,0.810,0.754,0.611,0.547 ]
th=[ 0.947, 0.947, 0.882, 0.809, 0.754, 0.611, 0.549]
fo=[0.946,0.944,0.878,0.815,0.760,0.613,0.542 ]
#fi=[0.944,0.945,0.886,0.809,0.745,0.612,0.546 ]
#si=[0.946,0.946,0.886,0.814,0.747,0.610,0.552 ]
#se=[0.945,0.942,0.880,0.801,0.752,0.611,0.550 ]
#ei=[0.946,0.945,0.885,0.805,0.750,0.610,0.544 ]
#ni=[0.946,0.945,0.888,0.816,0.745,0.609,0.550 ]
te=[ 0.944,0.945,0.885,0.805,0.757,0.610,0.559 ]



ax3.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax3.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax3.plot(x_values, fo,'m->',lw=3.5, markersize=12, label='10')
ax3.plot(x_values, te,'c-o',lw=3.5, markersize=12, label='40')
#ax3.plot(x_values, fi,'r->',lw=3.5, markersize=12, label='6')
#ax3.plot(x_values, si,'b-o',lw=3.5, markersize=12,label='7' )
#ax3.plot(x_values, se,'m-o',lw=3.5, markersize=12, label='8')
#ax3.plot(x_values, ei,'c-d',lw=3.5, markersize=12, label='9')
#ax3.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')

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
fig.legend( handles, labels, title='Effect of walk length t',loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=4, fontsize=25 )

 #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')

#plt.title('Performance of methods in event-to-user recommendation')
plt.savefig('link_prediction_len.pdf',  bbox_inches='tight',dpi=100)



plt.show()
