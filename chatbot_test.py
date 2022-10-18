import cream, numpy

cream.csys.clear()

def make_inputs(dataset:dict):
    input_words_list=list(dataset.values())
    result = []

    for values in input_words_list:
        for sentence in values:
            result += [i.replace("?","").replace(".","") for i in str(sentence).split(" ") if i.replace("?","").replace(".","") not in result]

    return result

def make_outputs(dataset:dict):
    keys = list(dataset.keys())

    return keys

def make_inputs_with_sentence(sentence:str, inputs:list):
    tokened = [i.replace("?","").replace(".","") for i in str(sentence).split(" ")]
    result = [0] * len (inputs)

    for token in tokened:
        result[inputs.index(token)] = 1

    return result



dataset = {
    "greeting":["hi", "hello", "nice to meet you"],
    "bye":["good bye", "see you again", "bye"],
    "name":["what's your name?", "what is your name?", "can you tell me your name?"],
    "age":["how old are you?", "please tell me your age?", "what's your age?"]
}

inputs = make_inputs(dataset)
len_inputs = len(inputs)
outputs = make_outputs(dataset)
len_outputs = len(outputs)

network = cream.network([len_inputs,5, 5,len_outputs],cream.functions.Linear,0.1)

#train

for epoch in range(100000):
    spaces = " "*(6-len(str(epoch)))
    print(f"epoch: {spaces}{epoch}    error: ",end=" ")
    total_error = 0

    for key in dataset:
        sentences = dataset[key]
        output_ = [0]*len(outputs)
        output_[outputs.index(key)] = 1

        for sentence in sentences:
            input_ = make_inputs_with_sentence(sentence, inputs)

            network.forward(input_)
            network.backpropgation(output_)
            total_error += sum(cream.functions.Error(network.activ[-1], output_))
    print(total_error)

    if (total_error < 0.1**8): break

while True:
    try:
        inp = str(input(" >>> ")).replace("?","").replace(".","")
        if (inp == "network"):
            cream.csys.out(network.weights, cream.csys.bcolors.OKGREEN)
            cream.csys.out(network.biases, cream.csys.bcolors.OKBLUE)
        else:
            input_ = make_inputs_with_sentence(inp, inputs)
            network.forward(input_)
            print(outputs[numpy.argmax(network.activ[-1])])
            cream.csys.out(network.activ[-1], cream.csys.bcolors.OKCYAN)
    except Exception as e:
        print(e)
        print(input_)
        print(network.activ[-1])