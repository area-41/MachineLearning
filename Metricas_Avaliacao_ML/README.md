### Desafio DIO - Machine Learning Specialist

##### Descrição do Desafio

Cálculo de Métricas de Avaliação de Aprendizado 

Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1). 

Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher de forma arbitraria, pois nosso objetivo é entender como funciona cada métrica.  


![image](https://user-images.githubusercontent.com/87396846/188651591-9f9821fd-e6dc-449a-b5d5-80db924672d2.png)


Tabela 1: Visão geral das métricas usadas para avaliar métodos de classificação. VP: verdadeiros positivos; FN: falsos negativos; FP: falsos positivos; VN: verdadeiros negativos; P: precisão; S: sensibilidade; N: total de elementos. 


o ARQUIVO com o Codigo abaixo:
<p><blockquote>
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


matrix = get_confusion_matrix(reais=real, preditos=predicao, labels=[1, 0])<br/>
print(matrix)<br/>
TP = matrix[0][0]<br/>
FP = matrix[0][1]<br/>
FN = matrix[1][0]<br/>
TN = matrix[1][1]<br/>

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

</blockquote></p><br/>

RESULTADO:

[[5 1]
 [2 5]]
 <br/>
Acuracia (TP+TN/(TP+FP+TN+FN)): 
5.384615384615385
<br/>
Sensibilidade (TP/(TP+FN)): 
0.7142857142857143
<br/>
Especificidade (TN/(TN+FP)): 
0.8333333333333334
<br/>
Precisão (TP/(TP+FP)): 
0.8333333333333334
<br/>
F-score (2x((precisaoXsensibilidade)/(precisao+sensibilidade)): 
0.7692307692307692
<br/>

Process finished with exit code 0

<br/><br/>


##### REFERENCIAS<br/>

*** codigo retirado de uma versao disponivel na internet:<br/>
https://medium.com/data-hackers/entendendo-o-que-%C3%A9-matriz-de-confus%C3%A3o-com-python-114e683ec509
<br/>
https://diegomariano.com/metricas-de-avaliacao-em-machine-learning/#Especificidade
