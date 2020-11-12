#기본 세팅
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#데이터 셋 구성
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 가중치 W, 편향 b 0으로 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#경사하강법 구현, lr: 학습률을 의미
optimizer = optim.SGD([W, b], lr=0.01)

#3000회 수행
nb_epochs = 3000

for epoch in range(nb_epochs + 1):

    #가설 세우기
    hypothesis = x_train * W + b
    
    #torch.mean으로 평균 구하기
    cost = torch.mean((hypothesis - y_train) ** 2)

    #gradient 0 초기화
    optimizer.zero_grad()
    #비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 최신화
    optimizer.step()

    #100번째마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost {:.6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))