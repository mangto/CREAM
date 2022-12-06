XOR = [
    [[0, 0], [0]],
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]]
]

Reverse = [
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[1, 1], [0, 0]],
    [[0, 0], [1, 1]]
]

Half = [
    [[0.05010025594726117], [0.025050127973630587]], 
    [[0.03928899313709585], [0.019644496568547926]], 
    [[0.08125174915598574], [0.04062587457799287]], 
    [[0.07755299078157905], [0.03877649539078952]],
    [[0.06223529182219187], [0.031117645911095937]], 
    [[0.017529385720956991], [0.008764692860478496]], 
    [[0.07397050972913042], [0.03698525486456521]], 
    [[0.07653077762657591], [0.038265388813287954]], 
    [[0.037314314544905136], [0.018657157272452568]], 
    [[0.0014902803684135768], [0.0007451401842067884]]
]

def unzip(dataset:list) -> list:
    return list(zip(*dataset))