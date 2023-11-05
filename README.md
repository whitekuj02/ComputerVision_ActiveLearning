# CV_Active_1

# 파일 소개

/Dataset/CUB2011.py Dataset module 데이터를 불러올 때 처리하는 module
/result gradCam.py에서 나온 결과값들을 저장 ( test 셋 각각에 대한 마지막 layer의 gradCAM 사진과 예측이 맞았는지에 대한 여부를 파일명으로 확인 가능)

base_code.py, hj.py hj2.py jione.py Seung_min_chung.py 각 팀원들이 실험해본 .py 파일 (최고기록이 아닐 수 있음.)

best_solution.py 가장 높게 정확도가 나온 코드 중 하나

best_model.pth 실험 결과의 weight를 저장함

best_model_94.295.pth 최고 기록 weight

gradCAM.py 특정 weight에 대한 각각의 test 결과 값과 마지막 layer의 GradCAM 확인

test.py 데이터 셋에 대한 분석을 할 때 사용했던 py

base_code.sh background로 base_code.py 돌리기 위한 shell script