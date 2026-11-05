import json,time
import torch
from paths import RESULTS
from photo_restoration.recognition_guided import initialize_frontend,load_frozen_classifier,file_hash
from photo_restoration.train_recognition_guided import _rects_tasks,_evaluate
from photo_restoration.continue_capacity import _atomic_json

torch.set_num_threads(4)
run=RESULTS/'photo-recognition-guided-100-v1';out=run/'epoch-025-rects';out.mkdir(exist_ok=True)
saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert saved['epoch']==25
classifier,classes=load_frozen_classifier('mps');frontend=initialize_frontend('mps');frontend.load_state_dict(saved['model'])
started=time.time()
metrics=_evaluate(frontend,classifier,_rects_tasks(classes),classes,out/'predictions.json')
previous=json.loads((RESULTS/'photo-restoration-rects-500-v1/result.json').read_text())
before=json.loads((run/'before-rects.json').read_text())['metrics']
result=dict(epoch=25,checkpoint_sha256=file_hash(run/'last.pt'),photos=500,n=4449,metrics=metrics,before_guided=before,
 binary=previous['metrics']['auto_binary'],previous_restorer=previous['metrics']['previous'],half_thickness_restorer=previous['metrics']['restored'],seconds=time.time()-started)
_atomic_json(out/'result.json',result)
lines=['# 25회 정답 지도 전처리 모델: ReCTS 500장 평가','','같은 사진 500장, 같은 정답 위치의 한자 4,449개. 현재 인식기 고정, ReCTS 학습 미사용. 추가 학습 없이 저장된 25회 체크포인트를 평가했다.','','|방식|Top-1|Top-5|Top-10|','|---|---:|---:|---:|']
for label,key in [('자동 반전·이진화','binary'),('이전 복원 100회','previous_restorer'),('절반 굵기 복원 100회 (이진화·크롭 연결)','half_thickness_restorer'),('정답 지도 시작 전 (연속값 연결)','before_guided'),('정답 지도 25회 (연속값 연결)','metrics')]:
 m=result[key];lines.append(f"|{label}|{m['top1']:.2%}|{m['top5']:.2%}|{m['top10']:.2%}|")
lines+=['','정답 지도 시작 전과 25회 모델은 동일한 연속값 출력→128px 축소→고정 인식기 경로를 쓴다. 이전 복원 실험과는 연결 방식이 달라 학습 효과는 시작 전 대비로 판단해야 한다.','', 'ReCTS는 반복 관찰한 진단 자료이며 독립 최종 시험 점수는 아니다. 지원하지 않는 한자·숫자·영문은 기존 평가와 동일하게 제외했다.']
(out/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
