## 2º trabalho da disciplina de IIA 2020/1
# Leonardo Rodrigues de Souza - 17/0060543

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn_pandas import CategoricalImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz,plot_tree

exclude = ['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)', 'Patient age quantile', 'SARS-Cov-2 exam result', 'Mycoplasma pneumoniae', 'Urine - pH']
class_column = 2
test_size = 0.25

def buildRandomForest(plotTree = 1, maxDepth = 3, number_of_trees = 5):
    # Leitura do arquivo de dados
    data = pd.read_excel('data/dataset.xlsx')

    ## Tratamento dos dados
    # Primeiro, é preciso definir quais colunas devem ser consideradas.
    # Depois, as colunas são divididas em predizores e classes. Os predizores serão usadas para tentar predizer as classes.
    predictors = data.iloc[:, 1:74]                 # Pega todas as colunas, exceto o ID do paciente, que não é relevante para o algoritmo
    predictors = predictors.drop(columns=exclude)   # Dropa as colunas que não devem ser consideradas 
    columns_predictors = predictors.columns         # Define o nome das colunas que serão consideradas
    classes = data.iloc[:,class_column].values      # Define as classes
    predictors = predictors.values

    ## Um problema frequente em massa de dados é a falta de dados, ou NaN
    # Para resolver este problema, a estratégia adotada foi preencher dados faltantes com o valor mais comum para aquela coluna
    simpleImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    predictors[:,0:15] = simpleImputer.fit_transform(predictors[:,0:15])
    predictors[:,32:38] = simpleImputer.fit_transform(predictors[:,32:38])
    predictors[:,40:48] = simpleImputer.fit_transform(predictors[:,40:48])
    predictors[:,49:64] = simpleImputer.fit_transform(predictors[:,49:64])

    ## Lida com os valores faltantes para as variáveis categórias.
    categoricalImputer = CategoricalImputer()
    predictors[:,15:21] = categoricalImputer.fit_transform(predictors[:,15:21])
    predictors[:,21:32] = categoricalImputer.fit_transform(predictors[:,21:32])
    predictors[:,38:40] = categoricalImputer.fit_transform(predictors[:,38:40])
    predictors[:,64:66] = categoricalImputer.fit_transform(predictors[:,64:66])
    predictors[:,48] = categoricalImputer.fit_transform(predictors[:,48])

    ## O sklearn randomForest somente aceita valores númericos como entrada. Portanto, é necessário a transformação
    # de valores categórias, string ou outros, para valores números.
    labelEncoder = LabelEncoder()
    for x in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 64, 65, 48]:
        predictors[:,x] = labelEncoder.fit_transform(predictors[:,x])

    ## Divide a base de dados entre valores para treinamento, normalmente 75%, e para teste, normalmente 25%.
    train_x,test_x,train_y,test_y = train_test_split(predictors, classes, test_size = test_size)

    ## Criação e treinamento da randomForest
    clf = RandomForestClassifier(max_depth=maxDepth, random_state=0)
    clf.fit(train_x, train_y)

    ## Teste da precisão da randomForest
    results = clf.predict(test_x)
    print (f'Precisão de {accuracy_score(test_y, results)}')

    if (plotTree == 1):
        ## Plota as 5 primeiras árvores geradas em uma imagem PNG
        fn = columns_predictors
        cn = np.unique(classes)
        fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
        for index in range(0, number_of_trees):
            plot_tree(clf.estimators_[index],
                        feature_names = fn, 
                        class_names=cn,
                        filled = True,
                        ax = axes[index]);

            axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
        fig.savefig('trees.png')
    elif (plotTree == 2):
        ## Imprime os 10 atributos mais importantes encontrados
        importance = clf.feature_importances_
        for i,v in sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]:
            print('Atributo: %s, Pontuação: %.5f' % (columns_predictors[i],v))

        ## Plota os 10 atributos mais importantes em uma imagem PNG
        columns_predictors = list(map(
            lambda i: columns_predictors[i[0]],
            sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]
        ))
        plot_importances = list(map(
            lambda x: x[1],
            sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]
        ))
        plt.rc('font', size=8)
        plt.bar(columns_predictors, plot_importances)
        plt.xticks(rotation='92')
        plt.tight_layout()
        plt.savefig("impostances.png", dpi=600)

def main():
    opt = -1
    while (opt < 1 or opt > 4):
        opt = int(input("Selecione a classe alvo:\n1 - Resultado do Covid\n2 - Se o paciente foi aceito em uma unidade de saúde\n3 - Se o paciente foi internado em uma unidade de tratamento semi-intensivo\n4 - Se o paciente foi internado em uma unidade de tratamento intensivo\nEntre com uma opção válida: "))
    
    if (opt == 2):
        class_column = 3
    elif (opt == 3):
        class_column = 4
    elif (opt == 4):
        class_column = 5

    maxDepth = int(input("Altura máxima da árvore: (padrão 3): "))
    printTree = -1
    while (printTree < 0 or printTree > 3):
        printTree = int(input("Selecione uma opção:\n0 - Não exportar imagem\n1 - Exportar imagem das 5 primeiras árvores\n2 - Exportar gráfico dos 10 melhores atributos\nSelecione uma opção: "))

    buildRandomForest(printTree, maxDepth)

if __name__ == "__main__":
    main()