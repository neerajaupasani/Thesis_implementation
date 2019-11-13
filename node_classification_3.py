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

tw=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]
th=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]
fo=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]

fi=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]
si=[0.74193,0.74193,0.73475,0.72572,0.70238,0.47265,0.47265]
se=[0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]




ei = [0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]
ni = [0.74193,0.74193,0.73475,0.72572,0.70238,0.51569,0.47265]
te=[0.74193,0.74193,0.73475,0.72572,0.70238,0.47265,0.47265]


ax1.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='2')
ax1.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='3' )
ax1.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='4')

ax1.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='5')
ax1.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax1.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
ax1.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax1.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
ax1.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')


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

tw=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
th=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
fo=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]

fi=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
si=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
se=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
ei=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
ni=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]
te=[0.44056,0.44056,0.44018,0.44402,0.45280,0.36803,0.36031]






ax2.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='2')
ax2.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='3' )
ax2.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='4')

ax2.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='5')
ax2.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax2.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
ax2.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax2.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
ax2.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')


ax2.set_title('Citeseer', fontsize=25)
ax2.set_ylabel('Macro-F1 score', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

tw=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327  ]
th=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327]
fo=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327]

fi=[0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327 ]
si=[0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327 ]
se=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327]
ei=[0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327 ]
ni=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327]
te=[ 0.54586,0.54586,0.54564,0.52929,0.51873,0.47797,0.44327]






ax3.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='2')
ax3.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='3')
ax3.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='4')

ax3.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='5')
ax3.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
ax3.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
ax3.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
ax3.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
ax3.plot(x_values, ni,'g-o',lw=3.5, markersize=12, label='10')

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
fig.legend( handles, labels, title='Effect of window size w', loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=9, fontsize=22 ) #bbox_to_anchor=(0.5, 0.99)
#fig.tight_layout()
#plt.savefig('./charts/Weibo-link-f1.pdf.pdf')


#plt.title('Performance of methods in event-to-user recommendation')
plt.savefig('node_classification_win.pdf',  bbox_inches='tight',dpi=100)



plt.show()
