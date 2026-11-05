**Agent Thinking Language: English** — All agent reasoning and internal thoughts must be in English.

## Code Design Principles

When writing code, don't just solve the problem—write **elegant code that integrates naturally with the existing codebase**:

- Follow SOLID principles.
- Adhere to Clean Architecture.
- **Fix at the root cause**: Fix where the problem originates, not where symptoms appear. If a model returns incompatible data, fix it at the model layer, not in every consumer.
- **Names reveal intent**: Function/variable names must clearly indicate scope and purpose. Avoid generic names like `Card`; use specific names like `DeckCard`.
- **Cohesion over convenience**: Keep related code together
- **Justify placement**: Before adding code to a file, ask "why here?" If the answer isn't clear, find a better location.
- **Minimize exposure**: Prefer private functions and limit exports. New code should integrate with existing patterns without introducing unnecessary new ones.

# Directory

- apps/dic, client, flutter
- tools/handwriting, 데이터 준비·학습·평가 도구
- resources/handwriting, 원본 데이터·공개 모델
- experiments/handwriting/results, 실험 결과·체크포인트
- docs/handwriting, 학습 계획
- vendor, 외부 프로젝트 서브모듈

# Reference

- 한자 인스타그램 영상 제작·게시 및 오전/저녁 게시 자동화는 저장소 스킬 `.agents/skills/hanja-instagram-reels/SKILL.md`를 읽고 따른다.

- bc-wd0: /Users/leo/flutter_workspace/bc-wd0
- visiondic: /Users/leo/ios_workspace/ocrdic
- simpledic: /Users/leo/aoo_workspace/simpledic
- shadowing: /Users/leo/aoo_workspace/shadowing
- shisen, 사천성: /Users/leo/aoo_workspace/shisen
- iconwiki: https://iconwiki.aoo.kr

- iOS 배포는 저장소 루트에서 `ao deploy`로 실행하며, 서명 빌드·Sentry 심볼 업로드·App Store Connect 업로드·심사 제출까지 완료한다.

- Flutter 개발 실행은 `aoo mac ensure` 또는 `aoo web ensure`로 작업 폴더별 세션을 재사용한다. `ensure` 재실행은 핫 리로드, 하단 메뉴 버튼은 핫 리스타트, `reload`는 핫 리스타트, `reload --hot`은 핫 리로드다. `stop --all`은 사용자 요청 시에만 실행한다.
