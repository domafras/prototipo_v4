# coding: utf-8
# utils.py: A common convention for utility modules containing general-purpose functions. 

from libs import *

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Valor Previsto')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão')
    plt.show()
    
def evaluate_model_by_feature(X_test, y_test, ft):

    unique_features = X_test[ft].unique()
    
    for feature in unique_features:
        # Filtrando as previsões e rótulos verdadeiros para a categoria específica
        indices = X_test[ft] == feature
        predictions = prediction[indices]
        true_labels = y_test[indices]

        # Avaliando o desempenho para a categoria específica
        print(f'\n{ft}: {feature}')
        print(classification_report(true_labels, predictions, zero_division=1))
        print(confusion_matrix(true_labels, predictions))