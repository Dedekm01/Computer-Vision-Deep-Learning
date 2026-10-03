import argparse, csv, json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap=argparse.ArgumentParser(); ap.add_argument('--raw',default='results/raw'); ap.add_argument('--figs',default='results/figures'); ap.add_argument('--csv',default='results')
A=ap.parse_args(); os.makedirs(A.figs,exist_ok=True)
order=['MobileNetV3-Large','ResNet-18','ResNet-50']; col={'MobileNetV3-Large':'#2a9d8f','ResNet-18':'#e9a23b','ResNet-50':'#4a6fa5'}
R={k:[json.load(open(f)) for f in sorted(glob.glob(f'{A.raw}/res_{k}_s*.json'))] for k in order}
ms=lambda k,f:(np.mean([f(r) for r in R[k]]),np.std([f(r) for r in R[k]]))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,3,figsize=(16,9)); ax=ax.ravel()
def bar(a,f,title,unit='',scale=1,fmt='{:.1f}'):
    v=[ms(k,f) for k in order]
    a.bar(order,[x[0]*scale for x in v],yerr=[x[1]*scale for x in v],color=[col[k] for k in order],capsize=4)
    for i,x in enumerate(v): a.text(i,x[0]*scale+x[1]*scale,fmt.format(x[0]*scale)+unit,ha='center',va='bottom',fontsize=10)
    a.set_title(title,loc='left',fontweight='bold'); a.tick_params(axis='x',labelsize=9)
bar(ax[0],lambda r:r['test_acc'],'Akurasi test (%)',scale=100); ax[0].set_ylim(90,101)
bar(ax[1],lambda r:r['test_f1'],'Macro F1 test (%)',scale=100); ax[1].set_ylim(90,101)
bar(ax[2],lambda r:r['test_loss'],'Loss test (lebih kecil lebih baik)',fmt='{:.3f}')
bar(ax[3],lambda r:r['latency_ms'],'Latensi CPU 1 thread, batch 1 (ms)')
bar(ax[4],lambda r:r['params_total_M'],'Parameter total (juta)')
for k in order:
    h=np.mean([r['hist']['val_acc'] for r in R[k]],0)*100; ax[5].plot(range(1,len(h)+1),h,marker='o',color=col[k],label=k)
ax[5].set_title('Akurasi validasi per epoch (%)',loc='left',fontweight='bold'); ax[5].set_xlabel('epoch'); ax[5].legend(frameon=False)
fig.suptitle(f'Transfer learning 3 model — {len(R[order[0]])} seed, mean ± std (11 kelas, input 128 px, CPU)',fontsize=14,fontweight='bold')
fig.tight_layout(); fig.savefig(f'{A.figs}/perbandingan_model.png',dpi=140)
# loss curves
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,k in zip(ax,order):
    for key,ls in [('train_loss','-'),('val_loss','--')]:
        a.plot(np.mean([r['hist'][key] for r in R[k]],0),ls,color=col[k],label=key)
    a.set_title(k,loc='left',fontweight='bold'); a.set_xlabel('epoch'); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{A.figs}/kurva_loss.png',dpi=140)
# confusion matrix sum over seeds
cl=R[order[0]][0]['classes']
fig,ax=plt.subplots(1,3,figsize=(18,5.5))
for a,k in zip(ax,order):
    cm=np.sum([r['cm'] for r in R[k]],0); a.imshow(cm,cmap='Blues')
    a.set_xticks(range(len(cl)));a.set_xticklabels(cl,rotation=60,ha='right',fontsize=8);a.set_yticks(range(len(cl)));a.set_yticklabels(cl,fontsize=8)
    for i in range(len(cl)):
        for j in range(len(cl)):
            if cm[i,j]: a.text(j,i,cm[i,j],ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else 'black')
    a.set_title(f'{k} (gabungan {len(R[k])} seed)',loc='left',fontweight='bold'); a.set_xlabel('prediksi'); a.set_ylabel('sebenarnya')
fig.tight_layout(); fig.savefig(f'{A.figs}/confusion_matrix.png',dpi=140)
# table
rows=[]
print(f"{'Model':20}{'Acc%':>14}{'F1%':>14}{'Loss':>14}{'Lat ms':>10}{'Param M':>9}{'Dilatih M':>10}{'Waktu s':>9}")
for k in order:
    a=ms(k,lambda r:r['test_acc']*100);f=ms(k,lambda r:r['test_f1']*100);l=ms(k,lambda r:r['test_loss']);t=ms(k,lambda r:r['latency_ms'])
    p=R[k][0]['params_total_M'];pt=R[k][0]['params_trained_M'];tt=ms(k,lambda r:r['train_time_s'])[0]
    print(f"{k:20}{a[0]:8.2f}±{a[1]:.2f}{f[0]:8.2f}±{f[1]:.2f}{l[0]:8.3f}±{l[1]:.3f}{t[0]:10.1f}{p:9.2f}{pt:10.2f}{tt:9.0f}")
    print('   per-seed acc:',[round(r['test_acc']*100,2) for r in R[k]])

with open(f'{A.csv}/ringkasan_model.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','n_seed','acc_mean','acc_std','f1_mean','f1_std','test_loss_mean','test_loss_std','latency_ms','params_total_M','params_trained_M','train_time_s'])
    for k in order:
        g=lambda fn:ms(k,fn)
        w.writerow([k,len(R[k])]+[round(x,4) for x in (*g(lambda r:r['test_acc']*100),*g(lambda r:r['test_f1']*100),*g(lambda r:r['test_loss']),g(lambda r:r['latency_ms'])[0],R[k][0]['params_total_M'],R[k][0]['params_trained_M'],g(lambda r:r['train_time_s'])[0])])
with open(f'{A.csv}/hasil_per_seed.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','seed','best_epoch','train_time_s','val_acc','test_acc','test_f1','test_loss','latency_ms'])
    for k in order:
        for r in R[k]: w.writerow([k,r['seed'],r['best_epoch'],round(r['train_time_s']),round(r['best_val_acc']*100,2),round(r['test_acc']*100,2),round(r['test_f1']*100,2),round(r['test_loss'],4),round(r['latency_ms'],1)])
