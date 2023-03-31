from sklearn.decomposition import PCA


def pca_of_gradients(gradients, num_components):
    pca = PCA(n_components=num_components)

    return pca.fit_transform(gradients)
