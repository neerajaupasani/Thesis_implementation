# import statements

import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 5))#figsize=(10, 4)
fig.subplots_adjust(hspace=0.1, wspace=0.3)
ax1.grid(b=True, linestyle=':')
ax2.grid(b=True, linestyle=':')
ax3.grid(b=True, linestyle=':')

x_values= [1, 2, 3, 4, 5, 6, 7]
my_xticks = ['-3', '-2', '-1', '0', '1', '2', '3']


y_values=[ 0.4,0.5,0.6, 0.7, 0.8]
my_yticks=[ '0.4', '0.5', '0.6', '0.7', '0.8']

tw=[0.79792,0.80249,0.80209,0.79574,0.78993,0.80651,0.80253]
th=[0.79979,0.79979,0.80398,0.80398,0.79979,0.80790,0.78551]
fo=[0.78755,0.79128,0.79174,0.79342,0.78781,0.80367,0.79850]

#fi=[0.77339,0.77339,0.77912,0.77912,0.77748,0.78664,0.78567]
#si=[0.80609,0.79882,0.79818,0.80132,0.79771,0.81465,0.82358]
#se=[0.80427,0.79657,0.79894,0.79990,0.79708,0.80960,0.81312]




#ei = [0.81596,0.82294,0.81680,0.81393,0.81556,0.81528,0.81034]
#ni = [0.79451,0.79204,0.79523,0.79095,0.79427,0.79986,0.80464]

te=[ 0.79016, 0.81051 ,0.80998 , 0.79349 , 0.81056 , 0.81536 , 0.82634]

ax1.plot(x_values, tw,'r-o',lw=3.5, markersize=12, label='20')
ax1.plot(x_values, th,'b-d',lw=3.5, markersize=12,label='30' )
ax1.plot(x_values, fo,'g-^',lw=3.5, markersize=12, label='10')

ax1.plot(x_values, te,'y-s',lw=3.5, markersize=12,label='40')
#ax1.plot(x_values, fi,'m->',lw=3.5, markersize=12, label='6')
#ax1.plot(x_values, si,'c-o',lw=3.5, markersize=12, label='7')
#ax1.plot(x_values, se,'r-s',lw=3.5, markersize=12,label='8')
#ax1.plot(x_values, ei,'b->',lw=3.5, markersize=12, label='9')

#ax1.plot(x_values, ni,'r-^',lw=3.5, markersize=12, label='10')


ax1.set_title('Cora' , fontsize=25)
ax1.set_ylabel('Macro-F1 score', fontsize=25)
ax1.set_xlabel(r'$\alpha $', fontsize=25)

ax1.set_xticks(x_values)
ax1.set_xticklabels(my_xticks, fontsize=25)

ax1.set_yticks(y_values)
ax1.set_yticklabels(my_yticks, fontsize=25)
ax1.tick_params(axis='both',labelsize=25)
#ax1.yaxis.tick_right()

tw=[0.53486,0.53518,0.53056,0.53264,0.53110,0.53904,0.54260]
th=[0.53703,0.53703,0.53703,0.53703,0.53535,0.53518,0.54344]
fo=[0.53179,0.53179,0.53179,0.53179,0.53179,0.53040,0.54847]

#fi=[0.52919,0.53061,0.52919,0.52919,0.52753,0.53361,0.54366]
#si=[0.53437,0.53437,0.53437,0.53259,0.52940,0.53863,0.54971]
#se=[0.54288,0.54288,0.54110,0.54110,0.53986,0.55393,0.56124]




#ei = [0.54354,0.54385,0.54306,0.54306,0.54385,0.54019,0.54695]
#ni = [0.53615,0.53777,0.53811,0.53675,0.53987,0.54716,0.54706]

te=[ 0.55319, 0.54921, 0.55692, 0.55958, 0.56020, 0.55961, 0.54809]

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
ax2.set_ylabel('Macro-F1 score', fontsize=25)
ax2.set_xlabel(r'$\alpha $', fontsize=25)

ax2.set_xticks(x_values)
ax2.set_xticklabels(my_xticks, fontsize=25)

ax2.set_yticks(y_values)
ax2.set_yticklabels(my_yticks, fontsize=25)
ax2.tick_params(axis='both',labelsize=25)

tw=[0.58361,0.59147,0.57696,0.57763,0.57854,0.56504,0.56331]
th=[0.59043,0.59511,0.59811,0.59913,0.60858,0.59725,0.60077]
fo=[0.59792,0.60453,0.60214,0.60046,0.59952,0.60753,0.59813]

#fi=[0.57628,0.58231,0.58263,0.58148,0.57920,0.59242,0.56655]
#si=[0.59959,0.59785,0.59993,0.60356,0.59772,0.61694,0.60823]
#se=[0.59837,0.57895,0.58659,0.59607,0.59426,0.59841,0.60457]

#ei = [0.59482,0.59794,0.59789,0.59505,0.59903,0.59450,0.58371]
#ni = [0.59756,0.59915,0.61003,0.60046,0.61570,0.60772,0.62081]

te=[ 0.56680, 0.57585, 0.57291, 0.55320, 0.56292, 0.52506, 0.47082]

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
ax3.set_ylabel('Macro-F1 score', fontsize=25)
ax3.set_xlabel(r'$\alpha $', fontsize=25)

ax3.set_xticks(x_values)
ax3.set_xticklabels(my_xticks, fontsize=25)

ax3.set_yticks(y_values)
ax3.set_yticklabels(my_yticks, fontsize=25)
ax3.tick_params(axis='both',labelsize=25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend( handles, labels, title='Effect of walk length t',loc='upper center' ,bbox_to_anchor=(0.42, 1.25) ,ncol=4, fontsize=25 )

plt.savefig('node_classification_len.pdf',  bbox_inches='tight',dpi=100)



plt.show()
