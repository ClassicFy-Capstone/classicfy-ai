"""Summarize verified protected-G final checkpoints and make a portable bundle."""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_selected_note_ae import dump
from embedding.selected_note_data import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();out=args.run
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    final=read('final_report.json');verification=read('verification.json');coverage=read('production_G_coverage.json')
    if final['status']!='complete' or verification['status']!='passed':raise ValueError('Unverified production')
    assert final['objective_mode']==coverage['objective_mode']=='g_only'
    assert coverage['training_clips']==final['clips']==5082 and coverage['reconstruction_clips']==0
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    summary={}
    for row,(era,info) in enumerate(final['eras'].items()):
        history=read(era+'/production/history.json');baseline=info['baseline']
        for col,(key,field) in enumerate([('geometry','score'),('reconstruction','loss')]):
            base=baseline[key][field];values=[1.]+[h['production_fit'][key][field]/base for h in history]
            axes[row,col].plot([0]+[h['epoch'] for h in history],values,marker='o',color='#156082')
            selected=info['production_epoch'];axes[row,col].scatter([selected],[values[selected]],marker='*',s=150,zorder=5)
            axes[row,col].axhline(1,color='gray',ls='--');axes[row,col].grid(alpha=.2)
            axes[row,col].set(title=era+(' | G objective' if col==0 else ' | Rec monitor'),xlabel='Epoch',ylabel='Ratio to all-data common AE')
            axes[row,col].ticklabel_format(axis='y',style='plain',useOffset=False)
        a,b=info['production_training_diagnostics'],baseline
        summary[era]=dict(clips=info['clips'],works=info['works'],selected_epoch=info['production_epoch'],
            max_epochs=info['max_epochs'],G_score_change_percent=100*(a['geometry']['score']/b['geometry']['score']-1),
            pair_MSE_change_percent=100*(a['geometry']['pair_mse']/b['geometry']['pair_mse']-1),
            reconstruction_change_percent=100*(a['reconstruction']['loss']/b['reconstruction']['loss']-1),
            normalized_z_drift=a['embedding']['z_anchor_mse'],preservation_passed=info['guard']['passed'],
            G_clips=coverage['eras'][era]['clips_entering_G_objective'],
            anchor_only_clips=coverage['eras'][era]['anchor_only_clips'],
            reload_max_error=verification['eras'][era]['reload_max_error'])
    fig.suptitle('Protected G final refits | full-data FIT diagnostics | no independent test',fontsize=14)
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    dump(out/'summary.json',dict(scope='Full-data fit diagnostics, not independent test',eras=summary))
    text='''보호 장치를 둔 G 중심 시대별 최종 모델

초기화: 전체 5082개로 복원 사전학습한 공통 AE final_all_5082.pt
목적함수: effective_lambda*G + 0.05*z_anchor + 0.01*head_shift
직접 복원 loss는 없음. 복원 기울기는 G 상한 계산과 보존 모니터링에 사용.
각 시대 모델은 독립적이며 같은 시대 안에서만 벡터 거리를 비교해야 함.

시대: baroque / classical_transition / romantic_impressionist
각 시대 final.pt는 score encoder, AE, residual head, 입력 scaler를 포함.
원본 실행 폴더의 production/last.pt는 재개용 상태이며, 배포 ZIP에는 선택된 final.pt만 포함.
해당 시대 전체 자료에서 보존 조건을 통과한 후보 중 G 목적값이 가장 낮은 epoch를 선택.

all_5082_era_embeddings.npz 필드
  perf_ids: 원본 대표구간 ID 문자열, 앞자리 0 유지
  composition_ids: 작품 ID
  era: 시대 키
  z_ae: 구간 윈도를 관측 음표 수로 가중평균한 64차원 AE 벡터
  z_style: head 적용 후 64차원 벡터
  e: L2 정규화한 z_style, 검색 및 비교에 사용
모든 벡터 배열 shape는 (5082,64). 시대별 embeddings.npz도 같은 필드를 포함.

사용 예
  import numpy as np
  f = np.load('all_5082_era_embeddings.npz', allow_pickle=False)
  mask = f['era'] == 'romantic_impressionist'
  vectors, ids = f['e'][mask], f['perf_ids'][mask]
  distances = 1 - vectors @ vectors[0]
  # distances[0]은 자기 자신. 작은 값일수록 가깝다.

PyTorch 체크포인트 추론 로더
  from embedding.era_note_style import load_era_checkpoint, aggregate_clip_latents
  model, head, state = load_era_checkpoint('baroque/final.pt')
  # 기존 note 전처리로 입력을 준비하고 state['scaler']를 사용한다.
  # model.encode_batch(batch)['z']를 관측 음표 수로 집계한 뒤 head(pooled)['e']를 사용한다.
  # G 프로필 자체는 추론 입력이 아니다.

모든5082구간을 encoder와 anchor에 사용하고 임베딩을 생성했다.
G 기준이나 배치 비교 상대가 부족한 구간은 직접 G 대신 anchor만 적용한다.
구간별 실제 G 참여 횟수는 production_G_coverage.json을 참조.

summary.json과 training_curves.png는 전체 학습 자료에서 계산한 진단이다.
이 최종 모델에는 독립 test가 없다. train 전용 평가 모델의 이전 test 결과는
final_report.json의 source_evaluation 경로에 있으며 최종 모델의 독립 성능이 아니다.
verification.json은 원본 해시, G 재생성, 동결 가중치, 구간 누락, e norm,
시대별 파일과 통합 파일 일치 및 모델 재로딩 일치를 검증한다.
'''
    for era,s in summary.items():text+='\n'+era+': '+json.dumps(s,ensure_ascii=False)+'\n'
    (out/'USAGE.txt').write_text(text,encoding='utf-8')
    files=[Path(n) for n in ['USAGE.txt','protocol.json','final_report.json','summary.json','verification.json',
        'production_G_coverage.json','metadata.json','training_curves.png','all_5082_era_embeddings.npz']]
    for era in final['eras']:
        files.extend(Path(era)/n for n in ['final.pt','embeddings.npz','g_production_scaler.json','production/history.json'])
    manifest=dict(format='protected-g-era-model-bundle-v1',clips=5082,objective_mode='g_only',
                  files={p.as_posix():file_hash(out/p) for p in files})
    dump(out/'artifact_manifest.json',manifest);files.append(Path('artifact_manifest.json'))
    archive=out/'protected_g_era_models_5082.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,name.as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Bundle CRC failed')
    dump(out/'completion.json',dict(status='complete',verification='passed',package=str(archive),
        package_sha256=file_hash(archive),summary=summary,embeddings_sha256=file_hash(final['embeddings'])))
    print('Protected G final bundle verified',archive,flush=True)


if __name__=='__main__':main()
