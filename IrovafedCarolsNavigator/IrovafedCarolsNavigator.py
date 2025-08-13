import sys
import os

import customtkinter as ctk
import webbrowser
import pyperclip
import datetime
import json
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
from PIL import Image


# Traduções embutidas (sem arquivo externo)
BUILTIN_TRANSLATIONS = {
    "pt": {
        # App
        "app_name": "Irovafed Carols Navigator",
        "btn_about": "Sobre",
        "about_title": "Sobre • ",
        "about_desc": (
            "Aplicativo interativo para auxiliar na escolha de modelos de Machine Learning, "
            "com base no tipo de problema (regressão, classificação, agrupamento, detecção de anomalias "
            "ou redução de dimensionalidade), necessidade de interpretabilidade e tamanho do dataset. "
            "Permite aplicar filtros avançados (robustez a outliers e exigência de normalização), "
            "acessar links de documentação oficial, copiar snippets prontos de código em Python e "
            "exportar os exemplos selecionados. Inclui ainda uma ferramenta integrada para validar "
            "datasets, sugerir objetivos de modelagem automaticamente e exibir comparações de desempenho "
            "por meio de métricas e gráficos."
        ),
        "about_created_by": "Criado por Caroline Brito Defavori | Data Science and Analytics",
        "about_brand": "Irovafed",
        "btn_ok": "OK",

        # Pergunta 1 (objetivo)
        "q1_title": "Qual é o objetivo do problema?",
        "q1_opt1": "Regressão | Prever um valor numérico contínuo (ex.: preço, temperatura, demanda).",
        "q1_opt2": "Classificação | Prever uma classe/etiqueta (ex.: aprovado/reprovado, spam/não-spam).",
        "q1_opt3": "Agrupamento | Descobrir grupos semelhantes sem rótulos prévios (análise exploratória).",
        "q1_opt4": "Detecção de Anomalias | Identificar pontos atípicos (fraudes, falhas, desvios).",
        "q1_opt5": "Redução de Dimensionalidade | Projetar/compactar features (exploração, visualização).",

        # AJUDAS DA PERGUNTA 1 (NOVO)
        "q1_help_reg": "Regressão: prever um valor numérico contínuo (ex.: preço, temperatura, demanda).",
        "q1_help_clf": "Classificação: prever uma classe/etiqueta (ex.: aprovado/reprovado, spam/não-spam).",
        "q1_help_cluster": "Agrupamento: descobrir grupos semelhantes sem rótulos prévios (análise exploratória).",
        "q1_help_anom": "Detecção de Anomalias: identificar pontos atípicos (fraudes, falhas, desvios).",
        "q1_help_dim": "Redução de Dimensionalidade: projetar/compactar features (exploração, visualização).",
        
        "btn_next": "Avançar",
        "btn_back": "Voltar",

        # Pergunta 2 (interpretabilidade)
        "q2_title": "Precisa de interpretabilidade alta?",
        "q2_opt1": "Sim",
        "q2_opt2": "Não",

        # Dataset
        "q_dataset_title": "Qual o tamanho do dataset?",
        "q_dataset_help": "Escolha uma estimativa do volume de dados disponível.",
        "dataset_small": "Pequeno",
        "dataset_medium": "Médio",
        "dataset_large": "Grande",
        "btn_recommendation": "Ver recomendações",

        # Resultado
        "result_title": "Recomendações para: ",
        "result_advanced_mode": "Modo avançado (filtrar)",
        "result_filter_outliers": "Robusto a outliers",
        "result_filter_normalization": "Não exigir normalização",
        "result_label_models": "Modelos sugeridos",
        "result_no_model": "Nenhum modelo atende aos filtros.",
        "card_docs": "Documentação",
        "card_copy": "Copiar exemplo",
        "card_copied": "Copiado!",
        "btn_export": "Exportar .py",
        "btn_restart": "Reiniciar",

        # Export
        "msg_export_success_title": "Exportação concluída",
        "msg_export_success_body": "Arquivo salvo em:\n",
        "msg_export_error_title": "Erro ao exportar",
        "msg_export_error_body": "Não foi possível salvar o arquivo. Detalhes: ",

        # Nomes/descrições dos modelos (Regressão)
        "reg_linear_nome": "Regressão Linear",
        "reg_linear_oque": "Modelo linear simples para prever valores contínuos.",
        "reg_linear_desc": "Rápido e interpretável; exige features escaladas e é sensível a outliers.",

        "ridge_nome": "Ridge Regression",
        "ridge_oque": "Linear com regularização L2 para lidar com multicolinearidade.",
        "ridge_desc": "Estabiliza coeficientes; requer normalização; sensível a outliers.",

        "lasso_nome": "Lasso",
        "lasso_oque": "Linear com L1 que pode zerar coeficientes irrelevantes.",
        "lasso_desc": "Faz seleção de variáveis; requer normalização; sensível a outliers.",

        "elasticnet_nome": "ElasticNet",
        "elasticnet_oque": "Combina L1+L2 equilibrando seleção e estabilidade.",
        "elasticnet_desc": "Flexível e robusta a multicolinearidade; requer normalização.",

        "knn_reg_nome": "kNN Regressor",
        "knn_reg_oque": "Prevê com base na média dos vizinhos mais próximos.",
        "knn_reg_desc": "Captura padrões locais; requer normalização; sensível a outliers.",

        "rf_reg_nome": "Random Forest Regressor",
        "rf_reg_oque": "Conjunto de árvores para prever valores contínuos.",
        "rf_reg_desc": "Robusto a outliers; funciona sem normalização; forte baseline.",

        "extra_reg_nome": "ExtraTrees Regressor",
        "extra_reg_oque": "Floresta de árvores extremamente aleatórias.",
        "extra_reg_desc": "Rápido e robusto; pouco tuning; não requer normalização.",

        "gbr_reg_nome": "Gradient Boosting Regressor",
        "gbr_reg_oque": "Boosting de árvores, sequencial, alta precisão.",
        "gbr_reg_desc": "Geralmente preciso; menos sensível a escala; pode sobreajustar.",

        "hgb_reg_nome": "HistGradientBoosting Regressor",
        "hgb_reg_oque": "Boosting por histogramas, mais rápido e escalável.",
        "hgb_reg_desc": "Indicado para grandes volumes; robusto; sem necessidade de normalização.",

        # Classificação — interpretabilidade "Sim"
        "log_reg_nome": "Regressão Logística",
        "log_reg_oque": "Modelo linear para classificação binária/multiclasse.",
        "log_reg_desc": "Interpretável; recomenda-se normalização das features.",

        "tree_class_nome": "Árvore de Decisão (Classificação)",
        "tree_class_oque": "Árvore única para separar classes.",
        "tree_class_desc": "Explicável e robusta a outliers; pode sobreajustar sem poda.",

        "gnb_nome": "Naive Bayes Gaussiano",
        "gnb_oque": "Classificador probabilístico simples e rápido.",
        "gnb_desc": "Bom baseline, especialmente em texto; não exige normalização.",

        "lda_nome": "Linear Discriminant Analysis (LDA)",
        "lda_oque": "Projeção linear para maximizar separação entre classes.",
        "lda_desc": "Interpretável; costuma exigir normalização; bom em dados lineares.",

        # Classificação — performance
        "rf_clf_nome": "Random Forest Classifier",
        "rf_clf_oque": "Floresta de árvores para classificação.",
        "rf_clf_desc": "Robusto a outliers; bom default; dispensa normalização.",

        "extra_clf_nome": "ExtraTrees Classifier",
        "extra_clf_oque": "Árvores extremamente aleatórias para classificação.",
        "extra_clf_desc": "Rápido e robusto; pouco tuning; sem normalização.",

        "gbc_clf_nome": "Gradient Boosting Classifier",
        "gbc_clf_oque": "Boosting de árvores focado em acurácia.",
        "gbc_clf_desc": "Alta performance tabular; menos sensível à escala.",

        "hgb_clf_nome": "HistGradientBoosting Classifier",
        "hgb_clf_oque": "Versão por histogramas, escalável.",
        "hgb_clf_desc": "Bom para datasets grandes; não requer normalização.",

        "svm_nome": "SVM (RBF)",
        "svm_oque": "Máquina de vetores de suporte para margens máximas.",
        "svm_desc": "Eficiente em datasets pequenos/médios; exige normalização.",

        "knn_clf_nome": "kNN Classifier",
        "knn_clf_oque": "Classifica pelos vizinhos mais próximos.",
        "knn_clf_desc": "Simples e eficaz; requer normalização; sensível a outliers.",

        "mlp_nome": "MLP Classifier",
        "mlp_oque": "Rede neural multicamada para classificação.",
        "mlp_desc": "Aprende padrões complexos; exige normalização e tuning.",

        # Agrupamento
        "kmeans_nome": "K-Means",
        "kmeans_oque": "Agrupamento por proximidade (centroides).",
        "kmeans_desc": "Rápido; requer normalização; sensível a outliers e k inicial.",

        "dbscan_nome": "DBSCAN",
        "dbscan_oque": "Agrupamento baseado em densidade.",
        "dbscan_desc": "Clusters arbitrários; robusto a outliers; sem normalização.",

        "agglo_nome": "Agglomerative Clustering",
        "agglo_oque": "Agrupamento hierárquico bottom-up (dendrograma).",
        "agglo_desc": "Bom para poucos milhares; recomenda-se normalização.",

        "gmm_nome": "Gaussian Mixture (GMM)",
        "gmm_oque": "Misturas de Gaussianas para clusters elípticos.",
        "gmm_desc": "Probabilístico; requer normalização; sensível a outliers.",

        "spectral_nome": "Spectral Clustering",
        "spectral_oque": "Clustering via grafos/espectro do Laplaciano.",
        "spectral_desc": "Captura formas não lineares; melhor em datasets pequenos; requer normalização.",

        # Redução de Dimensionalidade
        "pca_nome": "PCA",
        "pca_oque": "Projeção linear para reduzir dimensionalidade.",
        "pca_desc": "Útil para exploração/visualização; recomenda normalização.",

        # Detecção de Anomalias
        "iso_nome": "Isolation Forest",
        "iso_oque": "Isola outliers por divisões aleatórias.",
        "iso_desc": "Robusto e escalável; não exige normalização.",

        "lof_nome": "Local Outlier Factor (LOF)",
        "lof_oque": "Detecta outliers por densidade local.",
        "lof_desc": "Bom em estruturas locais; requer normalização.",

        "ocsvm_nome": "One-Class SVM",
        "ocsvm_oque": "Aprende a fronteira do 'normal'.",
        "ocsvm_desc": "Flexível mas sensível à escala; requer normalização.",
    },
    "en": {
        "app_name": "Irovafed Carols Navigator",
        "btn_about": "About",
        "about_title": "About • ",
        "about_desc": (
            "Interactive application to assist in selecting Machine Learning models based on "
            "the problem type (regression, classification, clustering, anomaly detection, or dimensionality reduction), "
            "interpretability requirements, and dataset size. It allows applying advanced filters "
            "(robustness to outliers and normalization requirements), accessing official documentation links, "
            "copying ready-to-use Python code snippets, and exporting the selected examples. "
            "It also includes an integrated tool to validate datasets, automatically suggest modeling objectives, "
            "and display performance comparisons through metrics and charts."
        ),
        "about_created_by": "Created by Caroline Brito Defavori | Data Science and Analytics",
        "about_brand": "Irovafed",
        "btn_ok": "OK",

        "q1_title": "What's the problem goal?",
        "q1_opt1": "Regression | Predict a continuous numeric value (e.g., price, temperature, demand).",
        "q1_opt2": "Classification | Predict a class/label (e.g., approved/denied, spam/not-spam).",
        "q1_opt3": "Clustering | Discover similar groups without prior labels (exploratory).",
        "q1_opt4": "Anomaly Detection | Find atypical points (fraud, failures, deviations).",
        "q1_opt5": "Dimensionality Reduction | Project/compact features (exploration, viz).",

        # QUESTION 1 HELPS (NEW)
        "q1_help_reg": "Regression: predict a continuous numeric value (e.g., price, temperature, demand).",
        "q1_help_clf": "Classification: predict a class/label (e.g., approved/denied, spam/not-spam).",
        "q1_help_cluster": "Clustering: discover similar groups without prior labels (exploratory).",
        "q1_help_anom": "Anomaly Detection: identify atypical points (fraud, failures, deviations).",
        "q1_help_dim": "Dimensionality Reduction: project/compact features (exploration, visualization).",

        "btn_next": "Next",
        "btn_back": "Back",

        "q2_title": "Do you need high interpretability?",
        "q2_opt1": "Yes",
        "q2_opt2": "No",

        "q_dataset_title": "What's the dataset size?",
        "q_dataset_help": "Pick an estimate of your available data volume.",
        "dataset_small": "Small",
        "dataset_medium": "Medium",
        "dataset_large": "Large",
        "btn_recommendation": "See recommendations",

        "result_title": "Recommendations for: ",
        "result_advanced_mode": "Advanced mode (filters)",
        "result_filter_outliers": "Robust to outliers",
        "result_filter_normalization": "No normalization required",
        "result_label_models": "Suggested models",
        "result_no_model": "No model matches the filters.",
        "card_docs": "Docs",
        "card_copy": "Copy example",
        "card_copied": "Copied!",
        "btn_export": "Export .py",
        "btn_restart": "Restart",

        "msg_export_success_title": "Export finished",
        "msg_export_success_body": "File saved at:\n",
        "msg_export_error_title": "Export error",
        "msg_export_error_body": "Couldn't save file. Details: ",

        # Model names (EN) — kept consistent keys
        "reg_linear_nome": "Linear Regression",
        "reg_linear_oque": "Simple linear model for continuous targets.",
        "reg_linear_desc": "Fast and interpretable; needs scaling and is outlier-sensitive.",

        "ridge_nome": "Ridge Regression",
        "ridge_oque": "Linear with L2 regularization for multicollinearity.",
        "ridge_desc": "Stabilizes coefficients; needs scaling; outlier-sensitive.",

        "lasso_nome": "Lasso",
        "lasso_oque": "Linear with L1 that can zero irrelevant coefficients.",
        "lasso_desc": "Feature selection; needs scaling; outlier-sensitive.",

        "elasticnet_nome": "ElasticNet",
        "elasticnet_oque": "Combines L1+L2 for balance.",
        "elasticnet_desc": "Flexible vs. multicollinearity; needs scaling.",

        "knn_reg_nome": "kNN Regressor",
        "knn_reg_oque": "Predicts by averaging nearest neighbors.",
        "knn_reg_desc": "Captures local patterns; needs scaling; outlier-sensitive.",

        "rf_reg_nome": "Random Forest Regressor",
        "rf_reg_oque": "Ensemble of trees for regression.",
        "rf_reg_desc": "Outlier-robust; no scaling; strong baseline.",

        "extra_reg_nome": "ExtraTrees Regressor",
        "extra_reg_oque": "Extremely randomized trees ensemble.",
        "extra_reg_desc": "Fast and robust; little tuning; no scaling.",

        "gbr_reg_nome": "Gradient Boosting Regressor",
        "gbr_reg_oque": "Sequential tree boosting for accuracy.",
        "gbr_reg_desc": "Accurate; less scale-sensitive; may overfit.",

        "hgb_reg_nome": "HistGradientBoosting Regressor",
        "hgb_reg_oque": "Histogram-based boosting, scalable.",
        "hgb_reg_desc": "Great for large datasets; robust; no scaling.",

        "log_reg_nome": "Logistic Regression",
        "log_reg_oque": "Linear model for binary/multiclass classification.",
        "log_reg_desc": "Interpretable; scaling recommended.",

        "tree_class_nome": "Decision Tree (Classification)",
        "tree_class_oque": "Single tree to separate classes.",
        "tree_class_desc": "Explainable; outlier-robust; can overfit.",

        "gnb_nome": "Gaussian Naive Bayes",
        "gnb_oque": "Simple, fast probabilistic classifier.",
        "gnb_desc": "Great baseline; no scaling required.",

        "lda_nome": "Linear Discriminant Analysis (LDA)",
        "lda_oque": "Linear projection maximizing class separation.",
        "lda_desc": "Interpretable; scaling often required.",

        "rf_clf_nome": "Random Forest Classifier",
        "rf_clf_oque": "Forest of trees for classification.",
        "rf_clf_desc": "Outlier-robust; no scaling.",

        "extra_clf_nome": "ExtraTrees Classifier",
        "extra_clf_oque": "Extremely randomized trees for classification.",
        "extra_clf_desc": "Fast, robust; little tuning; no scaling.",

        "gbc_clf_nome": "Gradient Boosting Classifier",
        "gbc_clf_oque": "Tree boosting focused on accuracy.",
        "gbc_clf_desc": "High performance; less scale-sensitive.",

        "hgb_clf_nome": "HistGradientBoosting Classifier",
        "hgb_clf_oque": "Histogram-based boosting, scalable.",
        "hgb_clf_desc": "Good for big data; no scaling.",

        "svm_nome": "SVM (RBF)",
        "svm_oque": "Support Vector Machine for max margins.",
        "svm_desc": "Good for small/medium; requires scaling.",

        "knn_clf_nome": "kNN Classifier",
        "knn_clf_oque": "Classifies by nearest neighbors.",
        "knn_clf_desc": "Simple and effective; needs scaling; outlier-sensitive.",

        "mlp_nome": "MLP Classifier",
        "mlp_oque": "Multilayer neural network for classification.",
        "mlp_desc": "Learns complex patterns; needs scaling and tuning.",

        "kmeans_nome": "K-Means",
        "kmeans_oque": "Centroid-based clustering.",
        "kmeans_desc": "Fast; needs scaling; sensitive to outliers.",

        "dbscan_nome": "DBSCAN",
        "dbscan_oque": "Density-based clustering.",
        "dbscan_desc": "Arbitrary clusters; outlier-robust; no scaling needed.",

        "agglo_nome": "Agglomerative Clustering",
        "agglo_oque": "Bottom-up hierarchical clustering.",
        "agglo_desc": "Good up to a few thousands; scaling recommended.",

        "gmm_nome": "Gaussian Mixture (GMM)",
        "gmm_oque": "Gaussian mixtures for elliptical clusters.",
        "gmm_desc": "Probabilistic; needs scaling; outlier-sensitive.",

        "spectral_nome": "Spectral Clustering",
        "spectral_oque": "Graph/Laplacian-based clustering.",
        "spectral_desc": "Captures non-linear shapes; small datasets; needs scaling.",

        "pca_nome": "PCA",
        "pca_oque": "Linear projection to reduce dimensionality.",
        "pca_desc": "Great for exploration/visualization; needs scaling.",

        "iso_nome": "Isolation Forest",
        "iso_oque": "Isolates anomalies via random splits.",
        "iso_desc": "Robust and scalable; no scaling required.",

        "lof_nome": "Local Outlier Factor (LOF)",
        "lof_oque": "Detects anomalies by local density.",
        "lof_desc": "Works on local structure; needs scaling.",

        "ocsvm_nome": "One-Class SVM",
        "ocsvm_oque": "Learns boundary of the 'normal' class.",
        "ocsvm_desc": "Flexible but scale-sensitive; needs scaling.",
    },
}

# Catálogo de recomendações
DADOS_RECOMENDACAO = {
    # ---------------- Regressão ----------------
    "regressao": {
        "titulo_key": "q1_opt1",
        "modelos": [
            {
                "id": "reg_linear",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
                "exemplo_codigo": "from sklearn.linear_model import LinearRegression\nmodelo = LinearRegression()\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "ridge",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
                "exemplo_codigo": "from sklearn.linear_model import Ridge\nmodelo = Ridge(alpha=1.0)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "lasso",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
                "exemplo_codigo": "from sklearn.linear_model import Lasso\nmodelo = Lasso(alpha=0.001)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "elasticnet",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
                "exemplo_codigo": "from sklearn.linear_model import ElasticNet\nmodelo = ElasticNet(alpha=0.001, l1_ratio=0.5)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "knn_reg",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
                "exemplo_codigo": "from sklearn.neighbors import KNeighborsRegressor\nmodelo = KNeighborsRegressor(n_neighbors=5)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "rf_reg",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
                "exemplo_codigo": "from sklearn.ensemble import RandomForestRegressor\nmodelo = RandomForestRegressor(n_estimators=200, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
            {
                "id": "extra_reg",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
                "exemplo_codigo": "from sklearn.ensemble import ExtraTreesRegressor\nmodelo = ExtraTreesRegressor(n_estimators=300, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
            {
                "id": "gbr_reg",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
                "exemplo_codigo": "from sklearn.ensemble import GradientBoostingRegressor\nmodelo = GradientBoostingRegressor(random_state=42)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
            {
                "id": "hgb_reg",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
                "exemplo_codigo": "from sklearn.ensemble import HistGradientBoostingRegressor\nmodelo = HistGradientBoostingRegressor(random_state=42)\nmodelo.fit(X_treino, y_treino)",
                "tamanho_dataset": ["Médio", "Grande"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
        ],
    },

    # ---------------- Classificação ----------------
    "classificacao": {
        "sim": {
            "titulo_key": "q2_opt1",
            "modelos": [
                {
                    "id": "log_reg",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
                    "exemplo_codigo": "from sklearn.linear_model import LogisticRegression\nmodelo = LogisticRegression(max_iter=1000, n_jobs=None)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": True},
                },
                {
                    "id": "tree_class",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
                    "exemplo_codigo": "from sklearn.tree import DecisionTreeClassifier\nmodelo = DecisionTreeClassifier(max_depth=5, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio"],
                    "tags": {"robusto_outliers": True, "requer_normalizacao": False},
                },
                {
                    "id": "gnb",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html",
                    "exemplo_codigo": "from sklearn.naive_bayes import GaussianNB\nmodelo = GaussianNB()\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": False},
                },
                {
                    "id": "lda",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
                    "exemplo_codigo": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nmodelo = LinearDiscriminantAnalysis()\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": True},
                },
            ],
        },
        "nao": {
            "titulo_key": "q2_opt2",
            "modelos": [
                {
                    "id": "rf_clf",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
                    "exemplo_codigo": "from sklearn.ensemble import RandomForestClassifier\nmodelo = RandomForestClassifier(n_estimators=300, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                    "tags": {"robusto_outliers": True, "requer_normalizacao": False},
                },
                {
                    "id": "extra_clf",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
                    "exemplo_codigo": "from sklearn.ensemble import ExtraTreesClassifier\nmodelo = ExtraTreesClassifier(n_estimators=300, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                    "tags": {"robusto_outliers": True, "requer_normalizacao": False},
                },
                {
                    "id": "gbc_clf",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
                    "exemplo_codigo": "from sklearn.ensemble import GradientBoostingClassifier\nmodelo = GradientBoostingClassifier(random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio"],
                    "tags": {"robusto_outliers": True, "requer_normalizacao": False},
                },
                {
                    "id": "hgb_clf",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
                    "exemplo_codigo": "from sklearn.ensemble import HistGradientBoostingClassifier\nmodelo = HistGradientBoostingClassifier(random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Médio", "Grande"],
                    "tags": {"robusto_outliers": True, "requer_normalizacao": False},
                },
                {
                    "id": "svm",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
                    "exemplo_codigo": "from sklearn.svm import SVC\nmodelo = SVC(kernel='rbf', probability=False)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": True},
                },
                {
                    "id": "knn_clf",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
                    "exemplo_codigo": "from sklearn.neighbors import KNeighborsClassifier\nmodelo = KNeighborsClassifier(n_neighbors=5)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Pequeno", "Médio"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": True},
                },
                {
                    "id": "mlp",
                    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
                    "exemplo_codigo": "from sklearn.neural_network import MLPClassifier\nmodelo = MLPClassifier(max_iter=300, random_state=42)\nmodelo.fit(X_treino, y_treino)",
                    "tamanho_dataset": ["Médio", "Grande"],
                    "tags": {"robusto_outliers": False, "requer_normalizacao": True},
                },
            ],
        },
    },

    # ---------------- Agrupamento ----------------
    "agrupamento": {
        "titulo_key": "q1_opt3",
        "modelos": [
            {
                "id": "kmeans",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
                "exemplo_codigo": "from sklearn.cluster import KMeans\nmodelo = KMeans(n_clusters=3, random_state=42)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "dbscan",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
                "exemplo_codigo": "from sklearn.cluster import DBSCAN\nmodelo = DBSCAN(eps=0.5, min_samples=5)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
            {
                "id": "agglo",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
                "exemplo_codigo": "from sklearn.cluster import AgglomerativeClustering\nmodelo = AgglomerativeClustering(n_clusters=3)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "gmm",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html",
                "exemplo_codigo": "from sklearn.mixture import GaussianMixture\nmodelo = GaussianMixture(n_components=3, random_state=42)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
            {
                "id": "spectral",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
                "exemplo_codigo": "from sklearn.cluster import SpectralClustering\nmodelo = SpectralClustering(n_clusters=3, assign_labels='kmeans', random_state=42)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
        ],
    },

    # ---------------- Detecção de Anomalias ----------------
    "anomalias": {
        "titulo_key": "q1_opt4",
        "modelos": [
            {
                "id": "iso",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html",
                "exemplo_codigo": "from sklearn.ensemble import IsolationForest\nmodelo = IsolationForest(contamination='auto', random_state=42)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": False},
            },
            {
                "id": "lof",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html",
                "exemplo_codigo": "from sklearn.neighbors import LocalOutlierFactor\nmodelo = LocalOutlierFactor(n_neighbors=20)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": True, "requer_normalizacao": True},
            },
            {
                "id": "ocsvm",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html",
                "exemplo_codigo": "from sklearn.svm import OneClassSVM\nmodelo = OneClassSVM(kernel='rbf', nu=0.1)\nmodelo.fit(X)",
                "tamanho_dataset": ["Pequeno", "Médio"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
        ],
    },

    # ---------------- Redução de Dimensionalidade ----------------
    "reducao_dim": {
        "titulo_key": "q1_opt5",
        "modelos": [
            {
                "id": "pca",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
                "exemplo_codigo": "from sklearn.decomposition import PCA\nmodelo = PCA(n_components=2)\nX_red = modelo.fit_transform(X)",
                "tamanho_dataset": ["Pequeno", "Médio", "Grande"],
                "tags": {"robusto_outliers": False, "requer_normalizacao": True},
            },
        ],
    },
}

def abrir_link(url): webbrowser.open_new_tab(url)
def copiar_para_clipboard(texto, botao, lang_manager):
    pyperclip.copy(texto); botao.configure(text=lang_manager.get("card_copied"), fg_color="green")
    botao.after(2000, lambda: botao.configure(text=lang_manager.get("card_copy"), fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))

class LanguageManager:
    def __init__(self):
        self.translations = BUILTIN_TRANSLATIONS
        self.current_lang = 'pt'
    def set_language(self, lang_code):
        if lang_code in self.translations: self.current_lang = lang_code
    def get(self, key, default=None):
        return self.translations.get(self.current_lang, {}).get(key, default if default else f"_{key}_")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = LanguageManager()
        self.current_frame_name = 'p1'
        self.app_version = "v1.0"
        self.title(f"{self.lang.get('app_name')} {self.app_version}")
        self.geometry("750x700"); self.minsize(700, 600)
        ctk.set_appearance_mode("System")
        try: ctk.set_default_color_theme("meu_tema.json")
        except FileNotFoundError: ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)

        self.objetivo_var = ctk.StringVar(); self.interpretabilidade_var = ctk.StringVar()
        self.dataset_var = ctk.StringVar(); self.advanced_mode_var = ctk.BooleanVar(value=False)
        self.outliers_var = ctk.BooleanVar(value=False); self.normalization_var = ctk.BooleanVar(value=False)
        self.language_var = ctk.StringVar(value="Português")

        self.frame_p1 = ctk.CTkFrame(self); self.frame_p2 = ctk.CTkFrame(self)
        self.frame_p_dataset = ctk.CTkFrame(self); self.frame_resultado = ctk.CTkFrame(self)

        top_controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_controls_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
        lang_switcher = ctk.CTkSegmentedButton(top_controls_frame, values=["Português", "English"], command=self.switch_language, variable=self.language_var)
        lang_switcher.pack(side="right", padx=(10, 0))
        self.theme_button = ctk.CTkButton(top_controls_frame, text="", width=30, command=self.alternar_tema)
        self.theme_button.pack(side="right")
        self.about_button = ctk.CTkButton(self, width=50, command=self.show_about_window)
        self.about_button.place(relx=0.0, rely=1.0, anchor="sw", x=20, y=-20)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.rebuild_ui()
        self.update_theme_button_icon()

        # >>> INTEGRAÇÃO DA NOVA GUIA (NÃO INTRUSIVA) <<<
        try:
            from tab_validator_recommender import attach_validator_tab
            attach_validator_tab(self)
        except Exception as e:
            print("Validador indisponível:", e)

    def on_closing(self):
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def alternar_tema(self):
        new_mode = "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.update_theme_button_icon()
        self.rebuild_ui()

    def update_theme_button_icon(self):
        if ctk.get_appearance_mode() == "Dark": self.theme_button.configure(text="☀️")
        else: self.theme_button.configure(text="🌙")

    def switch_language(self, value):
        self.lang.set_language('en' if value == 'English' else 'pt')
        self.title(f"{self.lang.get('app_name')} {self.app_version}")
        self.rebuild_ui()

    def rebuild_ui(self):
        for frame in [self.frame_p1, self.frame_p2, self.frame_p_dataset, self.frame_resultado]:
            for widget in frame.winfo_children(): widget.destroy()
        self.about_button.configure(text=self.lang.get("btn_about"))
        self.criar_frame_pergunta1(); self.criar_frame_pergunta2(); self.criar_frame_dataset()
        frame_map = {'p1': self.frame_p1, 'p2': self.frame_p2, 'p_dataset': self.frame_p_dataset, 'resultado': self.frame_resultado}
        if self.current_frame_name == 'resultado' and hasattr(self, 'last_title'):
            self.mostrar_recomendacao(self.last_title, self.base_recommendations)
        else:
            self.mostrar_frame(frame_map.get(self.current_frame_name, self.frame_p1))

    def show_about_window(self):
        about_window = ctk.CTkToplevel(self); about_window.title(self.lang.get("about_title") + self.lang.get("app_name"))
        about_window.geometry("550x300"); about_window.resizable(False, False)
        about_window.transient(self); about_window.grab_set()
        about_window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(about_window, text=self.lang.get("app_name"), font=('Helvetica', 20, 'bold')).pack(pady=(20, 10))
        ctk.CTkLabel(about_window, text=self.lang.get("about_desc"), wraplength=400).pack(pady=5)
        ctk.CTkFrame(about_window, height=2, fg_color="gray50").pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(about_window, text=self.lang.get("about_created_by")).pack()
        ctk.CTkLabel(about_window, text=self.lang.get("about_brand"), font=('Helvetica', 10, 'italic')).pack()
        ctk.CTkLabel(about_window, text=f"© {datetime.datetime.now().year}", font=('Helvetica', 10)).pack(pady=(10, 0))
        ctk.CTkButton(about_window, text=self.lang.get("btn_ok"), width=80, command=about_window.destroy).pack(pady=20)

    def dar_feedback_visual(self, widget):
        widget.configure(text_color="#E53935")
        self.after(1200, lambda: widget.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]))

    def update_wraplength(self, event, *labels): [label.configure(wraplength=event.width - 40) for label in labels]

    def criar_frame_pergunta1(self):
        frame = self.frame_p1; frame.grid_rowconfigure(0, weight=1); frame.grid_columnconfigure(0, weight=1)
        content = ctk.CTkFrame(frame, fg_color="transparent"); content.grid(row=0, column=0)
        self.label_pergunta1 = ctk.CTkLabel(content, text=self.lang.get('q1_title'), font=('Helvetica', 18, 'bold')); self.label_pergunta1.pack(pady=15, padx=20)

        op_frame = ctk.CTkFrame(content, fg_color="transparent"); op_frame.pack(pady=10, padx=20, anchor="w")
        opcoes = [
            ('q1_opt1', 'regressao'),
            ('q1_opt2', 'classificacao'),
            ('q1_opt3', 'agrupamento'),
            ('q1_opt4', 'anomalias'),
            ('q1_opt5', 'reducao_dim'),
        ]
        for key, val in opcoes:
            ctk.CTkRadioButton(op_frame, text=self.lang.get(key), variable=self.objetivo_var, value=val).pack(pady=6, anchor="w", padx=20)

        # (REMOVIDO) – antes havia um bloco que criava textos de ajuda duplicados abaixo das opções

        ctk.CTkButton(content, text=self.lang.get('btn_next'), command=self.processar_p1).pack(pady=20)

    def criar_frame_pergunta2(self):
        frame = self.frame_p2; frame.grid_rowconfigure(0, weight=1); frame.grid_columnconfigure(0, weight=1)
        content = ctk.CTkFrame(frame, fg_color="transparent"); content.grid(row=0, column=0)
        self.label_pergunta2 = ctk.CTkLabel(content, text=self.lang.get('q2_title'), font=('Helvetica', 18, 'bold')); self.label_pergunta2.pack(pady=15, padx=20)
        op_frame = ctk.CTkFrame(content, fg_color="transparent"); op_frame.pack(pady=10, padx=20, anchor="w")
        opcoes = [('q2_opt1', 'sim'), ('q2_opt2', 'nao')]
        for key, val in opcoes:
            ctk.CTkRadioButton(op_frame, text=self.lang.get(key), variable=self.interpretabilidade_var, value=val).pack(pady=10, anchor="w", padx=20)
        btn_frame = ctk.CTkFrame(content, fg_color="transparent"); btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text=self.lang.get('btn_back'), command=lambda: self.mostrar_frame(self.frame_p1)).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text=self.lang.get('btn_next'), command=self.processar_p2).pack(side="left", padx=10)

    def criar_frame_dataset(self):
        frame = self.frame_p_dataset; frame.grid_rowconfigure(0, weight=1); frame.grid_columnconfigure(0, weight=1)
        content = ctk.CTkFrame(frame, fg_color="transparent"); content.grid(row=0, column=0)
        self.label_dataset = ctk.CTkLabel(content, text=self.lang.get('q_dataset_title'), font=('Helvetica', 18, 'bold')); self.label_dataset.pack(pady=(15, 5), padx=20)
        ctk.CTkLabel(content, text=self.lang.get('q_dataset_help'), font=('Helvetica', 11, 'italic'), text_color="gray").pack(pady=(0, 15), padx=20)
        ctk.CTkSegmentedButton(
            content,
            values=[self.lang.get(k) for k in ['dataset_small', 'dataset_medium', 'dataset_large']],
            command=lambda v: self.dataset_var.set({"Pequeno": "Pequeno", "Médio": "Médio", "Grande": "Grande", "Small": "Pequeno", "Medium": "Médio", "Large": "Grande"}[v])
        ).pack(pady=10)
        btn_frame = ctk.CTkFrame(content, fg_color="transparent"); btn_frame.pack(pady=20)
        self.btn_voltar_dataset = ctk.CTkButton(btn_frame, text=self.lang.get('btn_back')); self.btn_voltar_dataset.pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text=self.lang.get('btn_recommendation'), command=self.processar_dataset).pack(side="left", padx=10)

    def mostrar_frame(self, frame):
        if frame == self.frame_p1: self.current_frame_name = 'p1'
        elif frame == self.frame_p2: self.current_frame_name = 'p2'
        elif frame == self.frame_p_dataset: self.current_frame_name = 'p_dataset'
        elif frame == self.frame_resultado: self.current_frame_name = 'resultado'
        for f in [self.frame_p1, self.frame_p2, self.frame_p_dataset, self.frame_resultado]: f.grid_forget()
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    def processar_p1(self):
        if not self.objetivo_var.get():
            self.dar_feedback_visual(self.label_pergunta1); return
        self.interpretabilidade_var.set(""); self.dataset_var.set("")
        if self.objetivo_var.get() == "classificacao":
            self.mostrar_frame(self.frame_p2)
        else:
            self.btn_voltar_dataset.configure(command=lambda: self.mostrar_frame(self.frame_p1))
            self.mostrar_frame(self.frame_p_dataset)

    def processar_p2(self):
        if not self.interpretabilidade_var.get():
            self.dar_feedback_visual(self.label_pergunta2); return
        self.btn_voltar_dataset.configure(command=lambda: self.mostrar_frame(self.frame_p2))
        self.mostrar_frame(self.frame_p_dataset)

    def processar_dataset(self):
        if not self.dataset_var.get():
            self.dar_feedback_visual(self.label_dataset); return

        objetivo = self.objetivo_var.get()
        if objetivo == "classificacao":
            dados = DADOS_RECOMENDACAO[objetivo][self.interpretabilidade_var.get()]
        else:
            dados = DADOS_RECOMENDACAO[objetivo]

        self.last_title = self.lang.get(dados['titulo_key'])
        modelos_filtrados = [m for m in dados['modelos'] if self.dataset_var.get() in m.get('tamanho_dataset', [])]
        self.mostrar_recomendacao(self.last_title, modelos_filtrados)

    def mostrar_recomendacao(self, titulo, lista):
        self.base_recommendations = lista; self.last_title = titulo
        self.advanced_mode_var.set(False); self.outliers_var.set(False); self.normalization_var.set(False)

        frame = self.frame_resultado; [w.destroy() for w in frame.winfo_children()]
        frame.grid_columnconfigure(0, weight=1); frame.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(frame, text=self.lang.get("result_title") + titulo, font=('Helvetica', 18, 'bold')).grid(row=0, column=0, pady=(20, 10), padx=20, sticky="ew")

        adv_frame = ctk.CTkFrame(frame, fg_color="transparent"); adv_frame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkSwitch(adv_frame, text=self.lang.get("result_advanced_mode"), variable=self.advanced_mode_var, command=self.toggle_advanced_mode).pack(side="left")
        self.advanced_filters_frame = ctk.CTkFrame(frame, fg_color="transparent"); self.advanced_filters_frame.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")
        ctk.CTkCheckBox(self.advanced_filters_frame, text=self.lang.get("result_filter_outliers"), variable=self.outliers_var, command=self.aplicar_filtros_avancados).pack(side="left", padx=10)
        ctk.CTkCheckBox(self.advanced_filters_frame, text=self.lang.get("result_filter_normalization"), variable=self.normalization_var, command=self.aplicar_filtros_avancados).pack(side="left", padx=10)

        self.scrollable_frame = ctk.CTkScrollableFrame(frame, label_text=self.lang.get("result_label_models"))
        self.scrollable_frame.grid(row=3, column=0, pady=10, padx=20, sticky="nsew"); self.scrollable_frame.grid_columnconfigure(0, weight=1)

        botoes_final = ctk.CTkFrame(frame, fg_color="transparent"); botoes_final.grid(row=4, column=0, pady=20)
        self.btn_exportar = ctk.CTkButton(botoes_final, text=self.lang.get("btn_export"), command=self.exportar_codigos)
        ctk.CTkButton(botoes_final, text=self.lang.get("btn_restart"), command=lambda: self.mostrar_frame(self.frame_p1)).pack(side="left", padx=10)

        self.toggle_advanced_mode(); self.aplicar_filtros_avancados()
        self.mostrar_frame(self.frame_resultado)

    def toggle_advanced_mode(self):
        if self.advanced_mode_var.get():
            self.advanced_filters_frame.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")
        else:
            self.advanced_filters_frame.grid_forget(); self.outliers_var.set(False); self.normalization_var.set(False)
        self.aplicar_filtros_avancados()

    def aplicar_filtros_avancados(self):
        lista_filtrada = self.base_recommendations
        if self.advanced_mode_var.get():
            if self.outliers_var.get():
                lista_filtrada = [m for m in lista_filtrada if m['tags'].get('robusto_outliers') is True]
            if self.normalization_var.get():
                lista_filtrada = [m for m in lista_filtrada if m['tags'].get('requer_normalizacao') is False]

        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.codigo_export = []

        if not lista_filtrada:
            ctk.CTkLabel(self.scrollable_frame, text=self.lang.get("result_no_model")).pack(pady=20)
        else:
            for rec in lista_filtrada:
                card = ctk.CTkFrame(self.scrollable_frame, border_width=1); card.pack(fill="x", padx=10, pady=10)
                model_id = rec['id']
                ctk.CTkLabel(card, text=self.lang.get(f"{model_id}_nome"), font=('Helvetica', 16, 'bold')).pack(anchor="w", padx=15, pady=(10, 5))
                l_oque = ctk.CTkLabel(card, text=self.lang.get(f"{model_id}_oque"), justify="left", font=('Helvetica', 12, 'italic')); l_oque.pack(anchor="w", padx=15, pady=(0, 10))
                l_desc = ctk.CTkLabel(card, text=self.lang.get(f"{model_id}_desc"), justify="left"); l_desc.pack(anchor="w", padx=15, pady=(0, 10))
                card.bind("<Configure>", lambda e, l1=l_oque, l2=l_desc: self.update_wraplength(e, l1, l2))
                acao_frame = ctk.CTkFrame(card, fg_color="transparent"); acao_frame.pack(fill="x", padx=15, pady=(5, 10))
                link = ctk.CTkLabel(acao_frame, text=self.lang.get("card_docs"), text_color="#6A8EDD", cursor="hand2"); link.pack(side="left", padx=(0, 20))
                link.bind("<Button-1>", lambda e, url=rec['url']: abrir_link(url))
                btn_copiar = ctk.CTkButton(acao_frame, text=self.lang.get("card_copy")); btn_copiar.pack(side="left")
                btn_copiar.configure(command=lambda code=rec['exemplo_codigo'], b=btn_copiar: copiar_para_clipboard(code, b, self.lang))
                self.codigo_export.append(f"# {self.lang.get(f'{model_id}_nome')}\n{rec['exemplo_codigo']}\n")

        if self.codigo_export: self.btn_exportar.pack(side="left", padx=10)
        else: self.btn_exportar.pack_forget()

    def exportar_codigos(self):
        if not hasattr(self, 'codigo_export') or not self.codigo_export: return
        caminho = asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py")])
        if caminho:
            try:
                with open(caminho, "w", encoding="utf-8") as f: f.write("".join(self.codigo_export))
                messagebox.showinfo(self.lang.get("msg_export_success_title"), self.lang.get("msg_export_success_body") + caminho)
            except Exception as e:
                messagebox.showerror(self.lang.get("msg_export_error_title"), self.lang.get("msg_export_error_body") + str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
