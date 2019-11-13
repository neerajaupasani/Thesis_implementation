# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['-3', '-2', '-1', '0', '1', '2', '3']


y_values=[ 0.7, 0.8, 0.9, 1.0]
my_yticks=[ '0.7', '0.8', '0.9', '1.0']

tw=[ 0.992, 0.991,0.988,0.987,0.987,0.987,0.987]
th=[ 0.989,0.988,0.990,0.990,0.989,0.989,0.989]
fo=[ 0.990, 0.989, 0.990, 0.989, 0.989, 0.989, 0.987]
#fi=[ 0.993,0.991,0.992,0.993,0.992,0.989,0.988]
#si=[ 0.992,0.992,0.992,0.992,0.991,0.991,0.987]
#se=[0.993,0.992,0.992,0.992,0.992,0.992,0.990 ]
#ei = [0.993,0.992,0.992,0.992,0.991,0.991,0.990 ]
#ni = [ 0.993,0.992,0.991,0.992,0.992,0.991,0.991]
te=[ 0.990, 0.993, 0.993, 0.990, 0.992, 0.990, 0.968]

ax1.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax1.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax1.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='10')
ax1.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='40')
#ax1.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
#ax1.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
#ax1.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
#ax1.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
#ax1.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')


ax1.set_title('Cora', fontsize=25)
ax1.set_ylabel('AUC score', fontsize=25)
ax1.set_xlabel(r'$\alpha $', fontsize=25)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=25)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=25)
ax1.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()

tw=[ 0.961,0.961,0.962,0.961,0.961,0.961,0.958]
th=[0.964,0.963,0.964,0.964,0.963,0.964,0.963 ]
fo=[0.963,0.964,0.963,0.964,0.963,0.964,0.963 ]
#fi=[ 0.963,0.963,0.964,0.963,0.963,0.964,0.961]
#si=[0.964,0.963,0.964,0.964,0.962,0.963,0.963 ]
#se=[0.964,0.964,0.962,0.964,0.962,0.963,0.962 ]
#ei = [0.963,0.963,0.963,0.965,0.962,0.964,0.963 ]
#ni = [0.963,0.963,0.964,0.963,0.962,0.964,0.963 ]
te=[ 0.965, 0.964, 0.964, 0.963, 0.965, 0.963, 0.958]

ax2.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax2.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax2.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='10')
ax2.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='40')
#ax2.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
#ax2.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
#ax2.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
#ax2.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
#ax2.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')


ax2.set_title('Citeseer' , fontsize=25)
ax2.set_ylabel('AUC score', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks, fontsize=25)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

tw=[0.990,0.988,0.981,0.988,0.982,0.977,0.978 ]
th=[0.987,0.989,0.991,0.994,0.992,0.987,0.973 ]
fo=[ 0.990,0.988,0.993,0.994,0.992,0.987,0.974]
#fi=[ 0.992,0.990,0.991,0.994,0.993,0.987,0.976]
#si=[0.994,0.992,0.992,0.994,0.993,0.987,0.974 ]
#se=[0.995,0.993,0.991,0.994,0.993,0.985,0.974 ]
#ei = [0.995,0.994,0.990,0.993,0.991,0.983,0.973 ]
#ni = [0.996,0.995,0.991,0.993,0.992,0.983,0.971 ]
te=[ 0.987, 0.988, 0.990, 0.983, 0.986, 0.839, 0.737]

ax3.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax3.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30')
ax3.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='10')
ax3.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='40')
#ax3.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
#ax3.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
#ax3.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
#ax3.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')
#ax3.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')


ax3.set_title('Wiki' , fontsize=25)
ax3.set_ylabel('AUC score', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)

ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks, fontsize=25)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
ax3.tick_params(axis='both',labelsize=25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend( handles, labels, title='Effect of walk length t',loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=4, fontsize=25 )

plt.savefig('link_prediction_pmi_len.pdf',  bbox_inches='tight',dpi=100)



plt.show()
