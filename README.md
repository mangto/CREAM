# CREAM (CoRy Enact Ai Model)

CREAM은 [**텐서플로우(tensorflow)**](https://github.com/tensorflow/tensorflow) 형태로 제작된 AI학습 모델입니다. 텐서플로우처럼 완벽하진 않지만, 대부분을 학습할 수 있습니다.


## XOR Algorithm 예제

### XOR Algorithm이란?
XOR Algorithm은 XOR 교체 알고리즘이라고 불립니다. 각 입력에 대한 결과는 다음과 같습니다.
|입력값|결과값| 
|-------|-------|
|0, 0|0|
|0, 1|1|
|1, 0|1|
|1, 1|0|

위의 예시를 보면 알 수 있듯, 입력 값이 서로 다면 1을, 같으면 0을 반환하는 것을 알 수 있습니다.

### CREAM 불러오기

    import cream

### 네트워크 만들기 (Generate Netowrk)

    network = cream.network(lrate = 0.01)
위와 같이 기본적인 네트워크를 만들 수 있습니다. **lrate인자를 변경**함으로써 **한 번에 얼마나 학습할 지** 정해줄 수 있습니다. lrate는 int  및 float 형태를 받습니다. **너무 크거나 너무 작은 lrate는 학습에 방해가 됩니다.**

### 레이어 추가 및 Compile (Add Layer & Compile)

    # network.add(Layer) -> None
	# cream.layer.Dense(size:int, activation=None, InputShape:int=None)
	
    network.add(cream.layer.Dense(5, cream.functions.ReLU, InputShape=2))
    network.add(cream.layer.Dense(1, crean,functions.ReLU))
    network.compile()
Dense 레이어는 가장 기본적인 네트워크 형태입니다. 뉴런들의 집합을 의미합니다. **network.add()을 통해 레이어를 추가**할 수 있으며, 가장 **첫 레이어에는 인풋 쉐입**을 넣어주어야 합니다.

 - size: 한 레이어 안의 뉴런 개수를 말하며, 정수형을 받습니다.
 - activation: activation function이라고도 부르며, sigmoid, ReLU, Leaky_ReLU 등이 있습니다.
 - InputShape: 첫 레이어에 넣어주는 인자로, 네트워크의 인풋 뉴런 개수를 정합니다. 

network.compile()은 네트워크를 완성시켜주는 것으로, **네트워크 사용을 위해서는 꼭 필요**합니다

### 데이터셋 불러오기 (Load dataset)
CREAM 안에는 몇가지 예제들이 있습니다. 그중 **XOR 데이터셋**을 불러오도록 하겠습니다.

    dataset = cream.dataset.XOR

다른 예제들로는 Reverse와 Half가 있습니다.

### 네트워크 사용하기 (Network Forward)

    network.forward(input)

### 역전파 학습 (Backpropagation)

    network.backward(target)

### 학습시키기 (Training)

    error  =  1
	epoch  =  0

	while (error  >  0.1**15  and  epoch  <=  10000):
		error  =  0
		for  data  in  dataset:
			network.forward(data[0])
			network.backward(data[1])
			error  +=  sum(cream.functions.Error(network.activ[-1], data[1]))
		cream.csys.out(f"epoch: {epoch:>6} | error: {error}", cream.csys.OKCYAN)
		epoch  +=  1
	cream.csys.stop()

위와 같이 **네트워크의 forward와 backpropagation을 반복하며 학습**시킬 수 있습니다.
CREAM에는 csys라는 모듈이 포함되어 있스며, 학습에 필요한 기능이 포함되어 있습니다. 이에 대해 밑에서 다시 설명하도록 하겠습니다.

## CSYS
CREAM 내에 **기본적으로 내장된 모듈**입니다. 기능들로는 **division, stop, error, clear, out**이 있습니다.

### division()

    cream.csys.division(length:int, Return=False)
콘솔(console)에 출력을 할 때에 공간 분리를 해주기 위함입니다. length는 4 이상의 정수열을 받습니다. Return 인자를 True로 변경하면 변수에 저장할 수 있습니다. 다음 예시를 참고하시기 바랍니다.

    import cream
	
	cream.csys.division(30)
	# +----------------------------+
	
	division = cream.csys.division(30, True)
	print(division)
	# +----------------------------+
#
### stop()

    cream.csys.stop(message:str="")
**코드를 잠시 중지**할 때에 사용되며, 에러를 고칠 때, 혹은 결과 값을 확인할 때 사용하면 유용합니다.

#
### error()

    cream.csys.error(message:str="", name:str="Unknown")
**코드에 에러가 발생했을 때에 사용**할 수 있습니다. 콘솔에 붉은 글씨로 출력이 됩니다. 에러가 출력된 이후에는 확인을 위해 코드가 중지되며(stop()과 동일하게) 이후 코드가 종료됩니다. 다음 예제를 확인하세요.

    import cream

	cream.csys.error("Description", "Error Name(Title)")
	# Error Occured (Error Name(Title)): Description

#
### clear()
```
cream.csys.clear()
```
콘솔의 글씨들을 모두 지웁니다.

#
### out()
```
cream.csys.out(message, color, bold:bool=False,underline:bool=False)
```
기존의 무채색 print에 색을 넣거나, 두껍게 할 수 있습니다.

# IDEA

![alt text](https://postfiles.pstatic.net/MjAyMjA3MTlfOTYg/MDAxNjU4MTY0MDM0OTcy.qDFvkYMrnnRLBpPzgGTMQt-dnaC-XxJLvLoqiM6rVesg.5VnCvtSvQ2_QyOlec59iNiXmrSxI2yd73cV2XyM7-pAg.JPEG.aka0115/KakaoTalk_20220719_020700741.jpg?type=w966)
![alt text](https://postfiles.pstatic.net/MjAyMjA3MTlfMjA0/MDAxNjU4MTY2NTUyMTY5.MiJpLbySyfZypTxtKwFTuB5dR_JswpWcB22jkf-B_aEg.1JTLRsl0VCxgV-oDl_4yVvWskoU3CNLErbR-3iCAMrQg.JPEG.aka0115/KakaoTalk_20220719_024849967.jpg?type=w966)
![alt text](https://postfiles.pstatic.net/MjAyMjA3MTlfMTI4/MDAxNjU4MTk5NTI3Njkz.e9lyl50oNK4BSqNUmEY3TA6R_kUaLs9wmByScUOiqIUg.Lbi46U5RiAVZG9p8fm4QKt6TLexjYjrWTqv3BPkfsuUg.JPEG.aka0115/KakaoTalk_20220719_115803300.jpg?type=w966)
## found rule
![alt text](https://postfiles.pstatic.net/MjAyMjA3MTlfMjU2/MDAxNjU4MjE4OTcwMDc1.qyneHM0mzXU61LIiIgnfBK2BpLVtREd83k47lMe5hGQg.qQPVUQojdqblOlw5NLO38fMaUNkM_5sy4B_DR2YX6RAg.JPEG.aka0115/KakaoTalk_20220719_172207256.jpg?type=w966)
