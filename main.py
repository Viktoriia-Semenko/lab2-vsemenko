import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


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

    return num_of_components


def reconstruct_pca_image(image_bw, num_of_components):
    image_pca = IncrementalPCA(n_components=num_of_components)  # кількість компонент
    new_image_components = image_pca.fit_transform(image_bw)  # нове зображ. з 48 компонентами
    image_reconstruction = image_pca.inverse_transform(new_image_components)  # повернення до початкової розмірності

    # Plotting the reconstructed image
    plt.figure(figsize=[12, 6])
    plt.imshow(image_reconstruction, cmap=plt.cm.gray)
    plt.show()


def reconstruction_dif_components(image_bw):

    dif_number_comp = [5, 15, 25, 75, 100, 170]

    plt.figure(figsize=[14, 8])

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        ipca = IncrementalPCA(n_components=dif_number_comp[i])
        image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
        plt.imshow(image_recon, cmap=plt.cm.gray)
        plt.title(f"Components: {dif_number_comp[i]}")

    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    plt.show()


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)

    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)

    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))  # діагоналізація матриці ключа
    decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)  # розшифрування вектора

    decrypted_message = ''. join([chr(int(np.round(num))) for num in decrypted_vector.real])  # конвертація у символ ASCII

    return decrypted_message


def main():
    image_raw = imread("rainbow.jpg")
    message = "Hello, World!"
    key_matrix = np.random.randint(0, 256, (len(message), len(message)))

    print("Виклик різних функцій:\n"
          "1 - Обчислення власних значень та власних векторів матриці\n"
          "2 - Виведення початкового зображення\n"
          "3 - Виведення чорно-білого зображення\n"
          "4 - Застосування PCA до чорно-білого зображення\n"
          "5 - Реконструкція зображення з обмеженою кількістю компонентів\n"
          "6 - Реконструкція зображення для різної кількості компонент\n"
          "7 - Розшифрування зашифрованого вектора\n")

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
    elif user_input == 5:
        image_bw = bw_image(image_raw)
        num = apply_pca_to_bw(image_bw)
        reconstruct_pca_image(image_bw, num)
    elif user_input == 6:
        image_bw = bw_image(image_raw)
        reconstruction_dif_components(image_bw)
    elif user_input == 7:
        encrypted_vector = encrypt_message(message, key_matrix)
        print(f"Original Message: {message}")
        print(f"Encrypted Message: {encrypted_vector}")

        decrypted_message = decrypt_message(encrypted_vector, key_matrix)
        print(f"Decrypted Message: {decrypted_message}")


if __name__ == "__main__":
    main()

