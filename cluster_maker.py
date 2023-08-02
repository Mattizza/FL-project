#Questa classe deve essere in grado di restituire un dizionario con key il nome del client e value i cluster associato
#Deve essere in grado di creare i cluster con varie tecniche
#Deve essere in grado di applicare la pca prima di fare la clusterizzazione
#Deve essere in grado, dato uno stile, di restituire a quale cluster appartiene
#Deve settare un parametro cluster id nel client
#Deve essere in grado di calcolare l'accuratezza del cluster sulla base di città o meteo??????
#Eventualmente deve essere in grado di fare una grid search sulle varie componenti della pca per decidere quale clustering è migliore

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class ClusterMaker():
    def __init__(self, styles_mapping, clients) -> None:
        self.styles_mapping = styles_mapping
        self.cluster_mapping = {}
        self.k_means_model = None
        self.num_clusters = len(self.cluster_mapping.keys())
        self.doPca = False
        self.clients = clients
        self.isFuzzy = False


    def flatten_styles(self):
        style_bank = self.styles_mapping["style"]
        styles_flat = np.array(style_bank).reshape(len(style_bank), -1)
        return styles_flat
    

    def cluster_styles(self):
        styles_flat = np.array(self.styles_mapping["style"]).reshape(len(self.styles_mapping["style"]), -1)
        
        if self.doPca:
            styles_flat = self.fit_transform_pipe_scaler_pca(styles_flat)

        model_list = []
        res_list = []
        score_list = []

        k_list = list(range(4, 20)) #Tests clustering from 4 to 19 centroids

        for k_size in k_list:
            model = KMeans(n_clusters=k_size, n_init=10).fit(styles_flat)
            model_list.append(model)
            res_list.append(model.labels_)
            score_list.append(silhouette_score(styles_flat, model.labels_))
        
        best_id = np.argmax(score_list)
        self.k_means_model = model_list[best_id]
        self.k_size = k_list[best_id]

        for cluster_id in range(self.k_size):
            self.cluster_mapping[cluster_id] = [self.styles_mapping["id"][i]
                                                for i, val in enumerate(res_list[best_id])
                                                if val == cluster_id]
        
        #Set the attribute cluster_id in each client
        for client in self.clients:
            for cluster_id in self.cluster_mapping.keys():
                if client.name in self.cluster_mapping[cluster_id]:
                    client.cluster_id = int(cluster_id)
                    break
    
    def fit_transform_pipe_scaler_pca(self, X, n_components = 3):
        self.pipe_scaler_pca = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components = n_components))])
        return self.pipe_scaler_pca.fit_transform(X)
    

    def assignClusterSingleImg(self, single_img_style):
        """
        Given the style of a single image, assign cluster
        """
        flatten_single_img = np.array(single_img_style).reshape(1, -1)
        
        if self.doPca:
            flatten_single_img = self.pipe_scaler_pca.transform(flatten_single_img)
        
        assigned_cluster_id = self.k_means_model.predict(flatten_single_img)
        
        return assigned_cluster_id




