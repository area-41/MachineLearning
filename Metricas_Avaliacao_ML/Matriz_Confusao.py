import numpy as np


real = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
predicao = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0]


def get_confusion_matrix(reais, preditos, labels):
    # não implementado
    if len(labels) > 2:
        return None

    if len(reais) != len(preditos):
        return None

    # considerando a primeira classe como a positiva, e a segunda a negativa
    true_class = labels[0]
    negative_class = labels[1]

    # valores preditos corretamente
    tp = 0
    tn = 0

    # valores preditos incorretamente
    fp = 0
    fn = 0

    for (indice, v_real) in enumerate(reais):
        v_predito = predicao[indice]

        # se trata de um valor real da classe positiva
        if v_real == true_class:
            tp += 1 if v_predito == v_real else 0
            fp += 1 if v_predito != v_real else 0
        else:
            tn += 1 if v_predito == v_real else 0
            fn += 1 if v_predito != v_real else 0

    return np.array([
        # valores da classe positiva
        [tp, fp],
        # valores da classe negativa
        [fn, tn]
    ])


matrix = get_confusion_matrix(reais=real, preditos=predicao, labels=[1, 0])
print(matrix)
TP = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]
TN = matrix[1][1]

print('Acuracia (TP+TN/(TP+FP+TN+FN)): ')
print(TP+TN/(TP+FP+TN+FN))

print('Sensibilidade (TP/(TP+FN)): ')
sensibilidade = TP/(TP+FN)
print(sensibilidade)

print('Especificidade (TN/(TN+FP)): ')
print(TN/(TN+FP))

print('Precisão (TP/(TP+FP)): ')
precisao = TP/(TP+FP)
print(precisao)

print('F-score (2x((precisaoXsensibilidade)/(precisao+sensibilidade)): ')
print(2*(precisao*sensibilidade/(precisao+sensibilidade)))