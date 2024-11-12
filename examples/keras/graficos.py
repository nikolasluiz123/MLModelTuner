from matplotlib import pyplot as plt


def plotar_resultados(history, fig_file_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    intervalo_epocas = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(intervalo_epocas, acc, 'r', label='Acurácia do Treino')
    plt.plot(intervalo_epocas, val_acc, 'b', label='Acurácia da Validação')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)

    plt.plot(intervalo_epocas, loss, 'r', label='Perda do Treino')
    plt.plot(intervalo_epocas, val_loss, 'b', label='Perda da Validação')
    plt.legend(loc='upper right')

    plt.savefig(f'{fig_file_name}.svg', format='svg')

    plt.show()