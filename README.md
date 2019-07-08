## Q-project: 문법 빈칸 문제 자동 출제

### Overview
- 영문법 빈칸 객관식 문제 출제 AI 훈련 모형
- 각 개별 문장 내 빈 칸 출제 대상 어휘를 1, 그 밖의 어휘를 0으로 분류하는 binary classification model

### 요구사항
- Tensorflow 2.0, NLTK, CUDA, Python 3 

### 파일
- [Download Glove Pre-trained Embeddings(glove.6B.300d.txt)](nlp.stanford.edu/data/glove.6B.zip): 단어 임베딩
- (v2)toeic_part5_question_data.csv: 훈련용 문제/문장 데이터
- GloVe_preprocessing.py: 전처리
- Group_data.py: GMM을 이용한 문장 분류
- metrics.py: Tensorflow 2.0 / Keras용 F1 / Precision / Recall
- Baseline_padding_all(10-folds).py: Bidirectional CuDNNLSTM 실험 모형 1. 전체 문장 벡터에 zero padding을 적용하여 같은 길이로 입력
- Baseline_padding_varying_size(10-folds).py: Bidirectional CuDNNLSTM 실험 모형 2. 현시점에서 masking이 지원되지 않는 CuDNN 라이브러리를 활용하면서 0값의 영향을 줄이기 위해 문장을 일정 길이 단위별로 분류하여 배치별로 서로 다른 길이의 zero padding을 적용
- Baseline_postag_random_embedding.py: 문법 요소를 이용한 분류를 위해 GloVe를 이용한 word embedding 대신 NLTK를 활용, 각 문장의 token을 POS-tag로 바꾼 뒤 이를 random embedding 하여 훈련하는 실험 모형
- Matrix_factorization.py: 문장-단어를 각각 축으로 하는 문장별 출제 단어의 matrix를 만들어서 low-rank factorization한 뒤, 문장의 hidden feature를 문장별 Bidirectional CuDNNGRU Training Vector로 연결하는 실험 모형

### 문제 (~19/07/07)
1. 출제 패턴의 다양성을 잡아내기에 Sample data가 부족한 것으로 추정(Training Set과 Validation Set의 Distribution 차이가Overfitting의 원인으로 여겨짐)
> - Training Set의 Accuracy/Recall/Precision 모두 1에 근접하나 Validation Set Accuracy는 0.95+, Recall은 0.2+, Precision은 0.3+ 정도에서 나아지지 않음
> - Training loss가 감소할수록 Validation loss가 증가
> - 실험결과 Network를 깊게 만드는 것은 효과가 없음 
> - Regularization은 학습속도를 느리게 할 뿐 epoch가 진행되면 없을 때와 유사한 결과에 도달

2. 모형은 문법적 패턴을 훈련하는가? 아니면 문장별로 출제되는 단어를 암기하는가?
> - Training Set의 target values를 shuffle한 다음 train한 실험 결과, 같은 Epoch에서 다음과 같은 결과를 보임
> 
>> **정상 set**: loss: 1.0038e-04 – accuracy: 1.0000 - recall: 0.9995 - precision: 0.9995  - val_loss: 0.3165 - val_accuracy: 0.9664 - val_recall: 0.2406 - val_precision: 0.2628
>> **실험 set**: loss: 0.0042 – accuracy: 0.9992 - recall: 0.9738 - precision: 0.9921  - val_loss: 0.4164 - val_accuracy: 0.9573 - val_recall: 0.0543 - val_precision: 0.0582
> - Training Set에서 metrics가 일정 수준 이상으로 올라간다는 것은 모형이 일정 수준은 답 자체를 외우고 있음을 시사
> - 하지만 Training Set의 metrics에도 실험/대조 사이에  약간이나마 차이가 있고, Validation Set의 metrics는 확연한 차이를 보인다. 이는 Training Set과 Validation Set 사이에 공유되는 **어떤 일반화될 수 있는 요소**가 Shuffle을 통해 사라졌음을 암시

3. Sample로 사용되는 TOEIC part 5의 문제들은 크게 문법을 묻는 문제와 적합한 표현을 찾는 문제로 나뉨. 이와 같은 특성이 문제의 보기 선택지에 드러난다는 점을 이용하여 지표들을 활용, GMM 모형으로 문제를 비지도학습, 여러 group들로 분류할 경우 일부 group에서는 Validation Recall이 0.5 수준으로 올라간다. 주로 문법 보다는 알맞은 표현을 묻는 문제에 해당되며, 해당 group에서 metrics의 상승이 관찰되는 원인으로는 모형이 답안을 암기하는 능력이 영향을 줄 가능성과, 각 group별 문제의 수가 줄면서 distribution도 좁혀졌을 가능성 정도가 고려됨.

4. 위 결과를 종합해보면, 실험에 사용되는 모형은 문장별 출제 단어를 기억하거나, 패턴을 어느정도 잡아내기에는 충분한 수의 parameter를 갖고 있는 것으로 보임. 문제는 근본적인 sample 확보의 한계. 실제로 sample을 충분히 확보해서 문제를 해결하기에는 대규모의 비용이 필요하기 때문에 생산성 면에서 비효과적임. 전처리나 기타 우회적인 모형의 개선을 통해 확률을 높일 필요가 있음.
