import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)  # повертає tuple що складається з вектора та масиву

    print("Власні значення матриці: ")
    print(eigenvalues)

    print("Власні вектори матриці: ")
    print(eigenvectors)

    for i in range(len(eigenvalues)):
        eig_values = eigenvalues[i]
        eig_vectors = eigenvectors[:, i]
        left_side = np.dot(matrix, eig_vectors)
        right_side = eig_values * eig_vectors

        print(f"\nПеревірка для власного значення {eig_values}:")
        print("A⋅v:")
        print(left_side)
        print("λ⋅v:")
        print(right_side)

        if np.allclose(left_side, right_side):
            print("Рівність A⋅v=λ⋅v виконується")
        else:
            print("Рівність A⋅v=λ⋅v не виконується")


def initial_image(image_raw):
    height, width, color_channels = image_raw.shape
    print(f"Вектор: {image_raw.shape}")
    print(f"Розміри зображення: {width} пікселів ширини та {height} пікселів висоти")
    print(f"Кількість основних каналів кольорів: {color_channels}")

    plt.figure(figsize=[12, 6])
    plt.imshow(image_raw)

    plt.show()


def bw_image(image_raw):
    image_sum = image_raw.sum(axis=2)  # сумування значення каналів RGB
    print(f"Розмір зображення: {image_sum.shape}")

    image_bw = image_sum / image_sum.max()
    print(f"Кількість каналів векторів: {image_bw.max()}")

    plt.figure(figsize=[12, 6])
    plt.imshow(image_bw, cmap=plt.cm.gray)
    plt.show()

    return image_bw


def check_eig_expression():
    matrix = np.array([[5, 6], [7, 8]])
    eigenvalues_and_eigenvectors(matrix)


def apply_pca_to_bw(image_bw):
    pca = PCA()  # створення обʼєкту PCA
    pca.fit(image_bw)  # обчислює компоненти PCA

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100  # обчислюється накопичена дисперсія
    num_of_components = np.argmax(cumulative_variance > 95)  # знаходить мінімальну кількість компонент
    print(f"Кількість компонент, які необхідні для покриття 95% дисперсії: {num_of_components}")

    plt.figure(figsize=[12, 6])
    plt.xlabel('Головні компоненти')
    plt.ylabel('Накопичена дисперсія (%)')
    plt.axvline(x=num_of_components, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    plt.plot(cumulative_variance, linewidth=2)
    plt.show()


def main():
    image_raw = imread("rainbow.jpg")

    print("Виклик різних функцій:\n"
          "1 - Обчислення власних значень та власних векторів матриці\n"
          "2 - Виведення початкового зображення\n"
          "3 - Виведення чорно-білого зображення")

    user_input = int(input("Введіть номер: "))
    if user_input == 1:
        check_eig_expression()
    elif user_input == 2:
        initial_image(image_raw)
    elif user_input == 3:
        bw_image(image_raw)
    elif user_input == 4:
        image_bw = bw_image(image_raw)
        apply_pca_to_bw(image_bw)


if __name__ == "__main__":
    main()

