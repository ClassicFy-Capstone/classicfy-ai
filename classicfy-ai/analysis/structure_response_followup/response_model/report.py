"""Render the pilot's objectives, retrieval controls and one observed response."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import OUT,PREVIOUS,ViewData
from model import ResponseEncoder


def read(name,root=OUT):
    with (root/name).open() as f:return list(csv.DictReader(f))


def bars(ax,axis,methods,summary,controls=None):
    x=np.arange(len(methods));width=.36
    for offset,(role,color) in enumerate((('test','#2563eb'),('validation','#0f766e'))):
        values=[]
        for method,label,source in methods:
            if source=='shuffle':r=controls[role,'response_aux',method,'performance_order_permuted']
            else:r=summary[role,method]
            values.append(float(r[axis+'_same_work'] if source=='same_work' else r[axis]))
        rectangles=ax.bar(x+(offset-.5)*width,values,width,color=color,label=role)
        ax.bar_label(rectangles,fmt='%.1f',fontsize=9,padding=3)
    ax.axhline(50,color='#64748b',ls='--',lw=1);ax.set_ylim(0,100)
    ax.set_xticks(x,[label for method,label,source in methods],rotation=12,ha='right')
    ax.set_ylabel('Relative score (random expectation 50)');ax.spines[['top','right']].set_visible(False)


def main():
    torch.set_num_threads(2)
    modes=('reconstruction_only','response','response_aux','context_only')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,mode in zip(axes.ravel(),modes):
        for view,color in ((0,'#2563eb'),(1,'#0f766e')):
            rows=read(f'view{view}_{mode}/history.csv');training=json.loads((OUT/f'view{view}_{mode}/training.json').read_text())
            ax.plot([int(r['epoch']) for r in rows],[float(r['validation_loss']) for r in rows],label=f'view {"AB"[view]}',color=color)
            ax.scatter([training['best_epoch']],[training['best']['loss']],color=color,s=35)
        ax.set_title(mode.replace('_',' '));ax.set_xlabel('Epoch');ax.set_ylabel('Own validation objective')
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend();fig.suptitle('Small conditional CNN: validation selects checkpoints\nObjectives differ across panels; do not compare their absolute heights',fontsize=13)
    fig.savefig(OUT/'01_training.png',dpi=160);plt.close(fig)
    summary={(r['role'],r['method']):r for r in read('summary.csv')+read('summary.csv',PREVIOUS)}
    controls={(r['role'],r['mode'],r['kind'],r['control']):r for r in read('control_summary.csv')}
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    p=[('F_summary','Fixed summary','actual'),('P_summary','Phrase summary','actual'),('untrained_P','Untrained encoder','actual'),
       ('reconstruction_only_P','Reconstruction only','actual'),('response_aux_P','Response model','actual'),('context_only_P','Score context only','actual')]
    h=[('H_summary','Tension summary','actual'),('untrained_H','Untrained encoder','actual'),('reconstruction_only_H','Reconstruction only','actual'),
       ('response_aux_H','Response model','actual'),('context_only_H','Score context only','actual')]
    bars(axes[0],'E-PN',p,summary);bars(axes[1],'E-HL',h,summary)
    axes[0].set_title('Phrase shape, change amplitude removed (E-PN)');axes[1].set_title('Tension-response timing agreement (E-HL)')
    axes[0].legend();fig.suptitle('Same 329 recordings, 241 gallery, 45 test / 43 validation\nValidation selected checkpoints; test did not',fontsize=13)
    fig.savefig(OUT/'02_retrieval.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,kind,axis,title in ((axes[0],'P','E-PN','Phrase response'),(axes[1],'H','E-HL','Tension response')):
        methods=[('response_aux_'+kind,'Actual order','actual'),(kind,'Performance order\npermuted','shuffle'),
                 ('context_only_'+kind,'Trained context only','actual'),('response_aux_'+kind,'Same returned work,\nrandom recording','same_work')]
        bars(ax,axis,methods,summary,controls);ax.set_title(title+' controls ('+axis+')')
    axes[0].legend();fig.suptitle('Sequence sensitivity and recording-selection specificity are separate checks',fontsize=13)
    fig.savefig(OUT/'03_controls.png',dpi=160);plt.close(fig)
    # Fixed example: central phrase of the first test recording; no favorable-example selection.
    data=ViewData(0);owner=int(data.arrays['owner'][data.roles['test'][0]])
    frames=np.flatnonzero((data.arrays['owner']==owner)&(data.arrays['type']==1));frame=int(frames[len(frames)//2])
    checkpoint=torch.load(OUT/'view0_response_aux/best.pt',weights_only=True)
    model=ResponseEncoder();model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.no_grad():_,_,heads=model(data.x[frame:frame+1],data.context[frame:frame+1])
    predicted=heads['shape'].reshape(2,16).numpy();actual=data.arrays['shape'][frame]
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for channel,ax in enumerate(axes):
        ax.plot(np.arange(16),actual[channel],color='black',lw=2,label='Observed normalized differences')
        ax.plot(np.arange(16),predicted[channel],color='#2563eb',lw=2,label='Learned response head')
        ax.axhline(0,color='#94a3b8',lw=.8);ax.set_ylabel(('Tempo','Dynamics')[channel]+' delta / RMS')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend();axes[1].set_xlabel('Successive positions within the 17-point phrase')
    fig.suptitle('Reading an observed response: '+Path(data.keys[owner]).name+'\nFull observed input; this does not generate a uniquely correct performance from score alone',fontsize=12)
    fig.savefig(OUT/'04_observed_response.png',dpi=160);plt.close(fig)
    (OUT/'example.json').write_text(json.dumps(dict(key=data.keys[owner],frame=frame,span=data.arrays['spans'][frame].tolist(),selection='Central phrase of first test recording, view A'),indent=2)+'\n')
    # Family-level evidence, including old baselines, for scrutiny beyond aggregate scores.
    records=read('scores.csv')+read('scores.csv',PREVIOUS);family_rows=[]
    for role in ('test','validation'):
        for family in sorted(set(r['family_id'] for r in records if r['role']==role)):
            for method in sorted(set(r['method'] for r in records)):
                rs=[r for r in records if r['role']==role and r['family_id']==family and r['method']==method]
                if not rs:continue
                family_rows.append(dict(role=role,family_id=family,method=method,
                    **{a:float(np.mean([float(r[a]) for r in rs])) for a in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL')}))
    with (OUT/'per_family.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(family_rows[0]));w.writeheader();w.writerows(family_rows)
    print('Rendered four figures and family-level metrics')


if __name__=='__main__':main()
