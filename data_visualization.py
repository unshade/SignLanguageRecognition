import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

train_df = pd.read_csv('./data/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('./data/sign_mnist_test/sign_mnist_test.csv')


def plot_redondance_graph():
    # Créer un histogramme multicolore
    sns.countplot(x='label', hue='label', data=train_df)

    # Ajouter une légende
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Ajouter un titre et des étiquettes aux axes
    plt.title('Histogramme de la redondance des labels')
    plt.xlabel('Label')
    plt.ylabel('Fréquence')

    # Afficher le graphique
    plt.show()


def plot_n_images(n):
    y_train = train_df['label']

    # Dropping the label column
    train_df.drop('label', axis=1, inplace=True)

    x_train = train_df.values / 255

    # Reshaping the data from 1-D to 3-D as required through input by CNN's
    x_train = x_train.reshape(-1, 28, 28, 1)

    f, ax = plt.subplots(2, 5)
    f.set_size_inches(10, 10)
    k = 0
    for i in range(n // 5):
        for j in range(5):
            ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
            ax[i, j].set_title("Label : {}".format(y_train[k]))
            k += 1
        plt.tight_layout()

    plt.show()


plot_n_images(10)
