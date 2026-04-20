from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data
# load real data
def load_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=1)
    df_with_header = pd.read_csv(data_path, sep='\t', header=None)

    x_data, y_data = [], []
    for i in range(len(df.columns) // 2):
        x = df.iloc[:, i * 2].values    # time of m0 curve
        y = df.iloc[:, i * 2 + 1].values    # signal of m0 curve
        x_data.append(x[~np.isnan(x)])
        y_data.append(y[~np.isnan(y)])

    m0vals = np.array([
        float(df_with_header.iloc[0, i * 2].split(': ')[-1])
        for i in range(len(df.columns) // 2)
    ])
    print('m0 values:', m0vals)
    return x_data, y_data, m0vals


if __name__ == "__main__":
    data_path = '/Users/nataliaionescu/Desktop/AB42_project/protein-aggregation-fits/ph_6.5_no_1.4_no_1.1.tsv' 
    # x_data times will not be the same for all m0 curves since i normalized it by hand and eliminated parts of it inconsistently
    # which is why we need to interpolate to get a data matrix 
    x_data, y_data, m0vals = load_data(data_path)
    tend = max(x[-1] for x in x_data)
    t_axis = np.linspace(0, tend, 300)
    data_matrix = [np.interp(t, x, y) for t, x, y in zip([t_axis] * len(x_data), x_data, y_data)]
    data_matrix = np.array(data_matrix)
    # somehow this data should acknowledge that triplicates share m0 and all curves share kinetic parameters
    # and then fit how, to what and interpret how????/????

    #=====
    # PCA 
    #=====

    pca = PCA()
    pca.fit(data_matrix)
    scores = pca.fit_transform(data_matrix)
    print(scores)

    '''explained variance ratio by each principal component'''
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()

    for i in range(3):
        plt.plot(t_axis, pca.components_[i], label=f'PC{i+1}')
    plt.show()

    # project kinetic curves onto principal components
    scores = pca.transform(data_matrix)
    m0_unique = sorted(set(m0vals))
    palette = sns.color_palette('tab10', n_colors=len(m0_unique))
    color_map = {m0: palette[i] for i, m0 in enumerate(m0_unique)}

    for m0 in m0_unique:
        color = color_map[m0]
        indices = np.where(m0vals == m0)[0]
        for i in indices:
            plt.scatter(scores[i,0], scores[i,1], color=color_map[m0], label=f'{m0}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
    # remove triplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


