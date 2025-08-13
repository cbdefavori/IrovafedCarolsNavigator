# Validador de Datasets e Recomendador de Modelos
# Integração mínima: chame attach_validator_tab(self) no __init__ do App.

import os
import time
import threading
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score,
    mean_squared_error, mean_absolute_error,
    silhouette_score, calinski_harabasz_score
)

# Modelos supervisionados
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.svm import OneClassSVM

# Modelos não supervisionados
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

ALL_TASKS = [
    "Regressão",
    "Classificação",
    "Agrupamento",
    "Detecção de Anomalias",
    "Redução de Dimensionalidade",
]

# Helpers de validação de colunas
def _is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_period_dtype(s) or pd.api.types.is_timedelta64_dtype(s)

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _invalid_target_reason(s: pd.Series, n_rows: int) -> str | None:
    nunique = s.nunique(dropna=True)
    if nunique <= 1:
        return "constante"
    if nunique == n_rows:
        return "provável ID (valor único por linha)"
    if _is_datetime_series(s):
        return "datetime/tempo"
    if s.dtype == object:
        if nunique > max(50, int(0.2 * n_rows)):
            return "texto de alta cardinalidade"
    return None

def _auto_task_from_target(s: pd.Series) -> str:
    nunique = s.nunique(dropna=True)
    if _is_numeric_series(s):
        return "Regressão" if nunique > 15 else "Classificação"
    return "Classificação"


# Janela/Guia autocontida
class ValidatorRecommenderWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Validador de Datasets e Recomendador de Modelos")
        self.geometry("900x700")
        self.minsize(900, 650)
        self.transient(master)
        self.grab_set()

        # Estado
        self.df: pd.DataFrame | None = None
        self.target_name: str | None = None
        self.task_var = ctk.StringVar(value="Classificação")
        self.target_var = ctk.StringVar(value="Selecione...")

        # Layout base
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)  # área de resultados expande

        # Linha 0: Upload
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        top_frame.grid_columnconfigure(0, weight=0)
        top_frame.grid_columnconfigure(1, weight=1)

        self.btn_upload = ctk.CTkButton(top_frame, text="Upload do Dataset", command=self.on_upload)
        self.btn_upload.grid(row=0, column=0, padx=8, pady=8)

        self.lbl_file = ctk.CTkLabel(top_frame, text="Nenhum arquivo selecionado", anchor="w")
        self.lbl_file.grid(row=0, column=1, sticky="ew", padx=8)

        # Linha 1: Feedback
        self.feedback = ctk.CTkTextbox(self, height=90)
        self.feedback.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        self.feedback.configure(state="disabled")

        # Linha 2: Seleções (Target e Objetivo)
        pick_frame = ctk.CTkFrame(self)
        pick_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        for i in range(4):
            pick_frame.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(pick_frame, text="Selecione a Variável Alvo (Target):").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        self.opt_target = ctk.CTkOptionMenu(pick_frame, values=["Selecione..."], variable=self.target_var, command=self.on_target_change)
        self.opt_target.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        ctk.CTkLabel(pick_frame, text="Qual o Objetivo do Modelo?").grid(row=0, column=1, sticky="w", padx=8, pady=(8, 0))
        # callback para atualizar estado conforme objetivo
        self.opt_task = ctk.CTkOptionMenu(pick_frame, values=ALL_TASKS, variable=self.task_var, command=self.on_task_change)
        self.opt_task.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))

        # Linha 3: Ações
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=16, pady=4)
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)

        self.btn_analyze = ctk.CTkButton(actions, text="Gerar Análise", state="disabled", command=self.on_analyze)
        self.btn_analyze.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        self.progress = ctk.CTkProgressBar(actions, mode="indeterminate")
        self.progress.grid(row=0, column=1, padx=8, pady=8, sticky="e")
        self.progress.grid_remove()

        # Linha 4: Dicas (atualizado)
        tips = ctk.CTkLabel(
            self,
            text=(
                "Regras de validação do Target:\n"
                "• Remove colunas constantes, prováveis IDs (valor único por linha), datetime/tempo e texto de alta cardinalidade.\n"
                "• Detecção automática do objetivo: numérico contínuo → Regressão; categórico/poucos valores → Classificação.\n"
                "• Para Agrupamento, Detecção de Anomalias e Redução de Dimensionalidade NÃO é necessário Target."
            ),
            justify="left",
            text_color="gray"
        )
        tips.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))

        # Linha 5: Resultados (gráficos)
        self.results = ctk.CTkFrame(self)
        self.results.grid(row=5, column=0, sticky="nsew", padx=16, pady=(8, 16))
        self.results.grid_columnconfigure(0, weight=1)
        self.results.grid_columnconfigure(1, weight=1)
        self.results.grid_rowconfigure(0, weight=1)

        # Estado inicial coerente com objetivo atual
        self.on_task_change(self.task_var.get())

    # ------------- Utils UI -------------
    def _feedback(self, text: str, clear: bool = False):
        self.feedback.configure(state="normal")
        if clear:
            self.feedback.delete("1.0", "end")
        self.feedback.insert("end", text.rstrip() + "\n")
        self.feedback.configure(state="disabled")
        self.feedback.see("end")

    def _clear_results(self):
        for w in self.results.winfo_children():
            w.destroy()

    # ------------- Eventos -------------
    def on_upload(self):
        path = askopenfilename(
            title="Selecione o arquivo de dados",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("Todos", "*.*")]
        )
        if not path:
            return

        self.lbl_file.configure(text=os.path.basename(path))
        self._clear_results()
        self._feedback("Lendo arquivo...", clear=True)

        try:
            df = None
            _, ext = os.path.splitext(path)
            ext = ext.lower()

            if ext == ".csv":
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_csv(path, sep=";")
            elif ext in (".xlsx", ".xls"):
                try:
                    df = pd.read_excel(path)
                except Exception as e:
                    raise RuntimeError("Para ler Excel, instale o pacote 'openpyxl'. Erro: " + str(e))
            else:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_excel(path)

            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("Dataset vazio ou inválido.")

            self.df = df
            self._feedback("Leitura do arquivo realizada com sucesso.")
            self._feedback(f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
            self._feedback(f"Colunas identificadas: {list(df.columns)}")

            # Popular targets válidos
            valid_targets, invalid_map = [], {}
            n_rows = len(df)
            for col in df.columns:
                reason = _invalid_target_reason(df[col], n_rows)
                if reason is None:
                    valid_targets.append(col)
                else:
                    invalid_map[col] = reason

            if not valid_targets:
                self.opt_target.configure(values=["Nenhuma coluna elegível"])
                self.target_var.set("Nenhuma coluna elegível")
                # estado final será decidido por on_task_change (unsupervised pode rodar)
            else:
                self.opt_target.configure(values=valid_targets)
                self.target_var.set("Selecione...")
                self.btn_analyze.configure(state="disabled")

            if invalid_map:
                summary = "; ".join([f"{k} ({v})" for k, v in invalid_map.items()])
                self._feedback(f"Colunas removidas do menu de Target: {summary}")

            # aplica as regras de habilitar/desabilitar conforme objetivo
            self.on_task_change(self.task_var.get())

        except Exception as e:
            self.df = None
            self.opt_target.configure(values=["Selecione..."])
            self.target_var.set("Selecione...")
            self.btn_analyze.configure(state="disabled")
            self._feedback("Não foi possível ler o arquivo. Verifique o formato e a integridade do mesmo.")
            messagebox.showerror("Erro", f"Falha ao ler arquivo: {e}")

    def on_task_change(self, selected: str):
        """Ajusta UI quando o objetivo muda.
        Objetivos NÃO supervisionados podem rodar sem target."""
        obj = (selected or "").strip()
        unsupervised = {"Agrupamento", "Detecção de Anomalias", "Redução de Dimensionalidade"}
        is_unsup = obj in unsupervised

        # Desabilita/habilita seletor de Target
        self.opt_target.configure(state="disabled" if is_unsup else "normal")

        # Habilita o botão de análise conforme necessidade de target
        if self.df is not None and not self.df.empty:
            if is_unsup:
                self.btn_analyze.configure(state="normal")
                self.target_name = None
                # feedback visual coerente
                self.target_var.set("Nenhuma coluna elegível")
            else:
                current = self.target_var.get()
                if current not in ("Selecione...", "Nenhuma coluna elegível", "", None):
                    self.btn_analyze.configure(state="normal")
                else:
                    self.btn_analyze.configure(state="disabled")
        else:
            self.btn_analyze.configure(state="disabled")

    def on_target_change(self, selected: str):
        # nunca testar DataFrame como booleano
        if getattr(self, "df", None) is None or self.df.empty or selected in ("Selecione...", "Nenhuma coluna elegível"):
            self.btn_analyze.configure(state="disabled")
            return

        self.target_name = selected
        suggestion = _auto_task_from_target(self.df[self.target_name])
        # Sugerimos Reg/Clf; usuário pode mudar depois
        if suggestion in ("Regressão", "Classificação"):
            self.task_var.set(suggestion)

        self._feedback(f"Target selecionado: {selected} • Objetivo sugerido: {suggestion}")
        # Habilita botão se objetivo atual precisa de target
        if self.task_var.get() not in {"Agrupamento", "Detecção de Anomalias", "Redução de Dimensionalidade"}:
            self.btn_analyze.configure(state="normal")

    def on_analyze(self):
        # checagem explícita do DataFrame + regra de target somente para Reg/Clf
        if getattr(self, "df", None) is None or self.df.empty:
            return

        task = self.task_var.get()
        if task in ("Regressão", "Classificação") and not self.target_name:
            return

        self._clear_results()
        self.progress.grid()
        self.progress.start()
        self.btn_analyze.configure(state="disabled")

        thread = threading.Thread(target=self._run_analysis_safe, args=(task,), daemon=True)
        thread.start()

    # ------------- Pipeline de análise -------------
    def _run_analysis_safe(self, task: str):
        try:
            self._run_analysis(task)
            ok = True
            msg = "Análise concluída com sucesso!"
        except Exception as e:
            ok = False
            msg = f"Falha durante a análise: {e}"

        self.after(0, lambda: self._finish_analysis(ok, msg))

    def _finish_analysis(self, ok: bool, msg: str):
        self.progress.stop()
        self.progress.grid_remove()
        # Reaplica regra de habilitar botão conforme objetivo/target/dataset
        self.on_task_change(self.task_var.get())
        self._feedback(msg)
        if not ok:
            messagebox.showerror("Erro", msg)

    def _prep_features(self, use_target=True):
        """Prepara X, y (quando existir), com dummies e imputação simples."""
        df = self.df.copy()
        target = self.target_name if use_target else None

        if target and target in df.columns:
            y = df[target]
            X = df.drop(columns=[target])
        else:
            y = None
            X = df

        # One-hot para categóricas; depois imputação em numéricas
        X = pd.get_dummies(X, drop_first=True, dummy_na=True)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())
        return X, y

    def _run_analysis(self, task: str):
        results_main, results_aux = [], []

        if task == "Classificação":
            X, y = self._prep_features(use_target=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42,
                stratify=y if y.nunique() > 1 else None
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = [
                ("Regressão Logística", LogisticRegression(max_iter=1000)),
                ("KNN", KNeighborsClassifier(n_neighbors=5)),
                ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
                ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ]
            for name, model in models:
                if name in ("Regressão Logística", "KNN"):
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                results_main.append((name, acc))
                results_aux.append((name, rec))

            self._plot_bar(results_main, title="Classificação • Acurácia por Modelo", ylabel="Acurácia", col=0)
            self._plot_bar(results_aux, title="Classificação • Recall (macro) por Modelo", ylabel="Recall (macro)", col=1)

        elif task == "Regressão":
            X, y = self._prep_features(use_target=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = [
                ("Regressão Linear", LinearRegression()),
                ("Ridge", Ridge(alpha=1.0)),
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
                ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42)),
            ]
            for name, model in models:
                if "Regressão Linear" in name or "Ridge" in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                results_main.append((name, rmse))
                results_aux.append((name, mae))

            self._plot_bar(results_main, title="Regressão • RMSE por Modelo (menor é melhor)", ylabel="RMSE", col=0, lower_is_better=True)
            self._plot_bar(results_aux, title="Regressão • MAE por Modelo (menor é melhor)", ylabel="MAE", col=1, lower_is_better=True)

        elif task == "Agrupamento":
            # sem target — usa todo o X
            X, _ = self._prep_features(use_target=False)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            clusterers = [
                ("KMeans (k=3)", KMeans(n_clusters=3, n_init="auto", random_state=42)),
                ("Agglomerative (k=3)", AgglomerativeClustering(n_clusters=3)),
                ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
            ]
            for name, algo in clusterers:
                labels = algo.fit_predict(Xs)
                # Para silhouette/CH precisamos de pelo menos 2 clusters válidos (ignorando ruído -1)
                mask = labels != -1
                labels_eff = labels[mask] if mask.any() else labels
                X_eff = Xs[mask] if mask.any() else Xs
                if len(np.unique(labels_eff)) >= 2 and len(labels_eff) >= 2:
                    sil = float(silhouette_score(X_eff, labels_eff))
                    ch = float(calinski_harabasz_score(X_eff, labels_eff))
                else:
                    sil, ch = np.nan, np.nan

                results_main.append((name, sil))
                results_aux.append((name, ch))

            self._plot_bar(results_main, title="Agrupamento • Silhouette por Algoritmo (maior é melhor)", ylabel="Silhouette", col=0)
            self._plot_bar(results_aux, title="Agrupamento • Calinski-Harabasz (maior é melhor)", ylabel="CH Index", col=1)

        elif task == "Detecção de Anomalias":
            X, _ = self._prep_features(use_target=False)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            detectors = [
                ("Isolation Forest", IsolationForest(contamination="auto", random_state=42)),
                ("LOF", LocalOutlierFactor(n_neighbors=20, novelty=False)),
                ("One-Class SVM", OneClassSVM(kernel="rbf", nu=0.1)),
            ]
            for name, model in detectors:
                t0 = time.time()
                if name == "LOF":
                    labels = model.fit_predict(Xs)
                else:
                    model.fit(Xs)
                    labels = model.predict(Xs)
                elapsed = time.time() - t0
                outlier_rate = float((labels == -1).mean() * 100.0)
                results_main.append((name, outlier_rate))
                results_aux.append((name, elapsed))

            self._plot_bar(results_main, title="Anomalias • % de Outliers Detectados", ylabel="% Outliers", col=0)
            self._plot_bar(results_aux, title="Anomalias • Tempo de Execução (s) (menor é melhor)", ylabel="Tempo (s)", col=1, lower_is_better=True)

        elif task == "Redução de Dimensionalidade":
            X, _ = self._prep_features(use_target=False)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            variants = [
                ("PCA (2D)", 2),
                ("PCA (3D)", 3),
            ]
            for name, k in variants:
                pca = PCA(n_components=min(k, Xs.shape[1]))
                Z = pca.fit_transform(Xs)
                explained = float(np.sum(pca.explained_variance_ratio_))
                # Reconstrução para RMSE (menor é melhor)
                X_hat = pca.inverse_transform(Z)
                rmse = float(np.sqrt(np.mean((Xs - X_hat) ** 2)))
                results_main.append((name, explained))
                results_aux.append((name, rmse))

            self._plot_bar(results_main, title="Redução de Dimensionalidade • Variância Explicada", ylabel="Proporção", col=0)
            self._plot_bar(results_aux, title="Redução de Dimensionalidade • RMSE de Reconstrução (menor é melhor)", ylabel="RMSE", col=1, lower_is_better=True)

        else:
            raise ValueError(f"Objetivo não suportado: {task}")

    # ------------- Plotagem embutida -------------
    def _plot_bar(self, pairs, title: str, ylabel: str, col: int, lower_is_better: bool = False):
        """pairs: list[(name, value)]"""
        chart_frame = ctk.CTkFrame(self.results)
        chart_frame.grid(row=0, column=col, sticky="nsew",
                         padx=(8 if col == 1 else 0, 8 if col == 0 else 0), pady=8)
        self.results.grid_columnconfigure(col, weight=1)

        labels = [p[0] for p in pairs]
        # aceita NaN / None
        values = [float(p[1]) if p[1] is not None else float("nan") for p in pairs]

        fig = Figure(figsize=(5.6, 3.8), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, np.nan_to_num(values, nan=0.0))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=25, ha="right")

        # headroom no topo p/ não cortar anotações
        if len(values):
            finite_vals = [v for v in values if np.isfinite(v)]
            if finite_vals:
                vmax = max(finite_vals)
                vmin = min(0.0, min(finite_vals))
                top_pad = 1.15 if vmax > 1.0 else 1.05
                top = (vmax * top_pad) if vmax > 0 else 1.0
                ax.set_ylim(vmin, top)

        # Anota valores (ou "—" quando NaN)
        for bar, v in zip(bars, values):
            txt = f"{v:.3f}" if np.isfinite(v) else "—"
            ax.annotate(txt,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        if lower_is_better and any(np.isfinite(v) for v in values):
            best = min(v for v in values if np.isfinite(v))
            ax.axhline(best, linestyle="--", linewidth=1)

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# Integração mínima com o app
def _open_window_from_app(app):
    ValidatorRecommenderWindow(master=app)

def attach_validator_tab(app):
    try:
        btn = ctk.CTkButton(app, width=100, text="Validador", command=lambda: _open_window_from_app(app))
        btn.place(relx=0.0, rely=1.0, anchor="sw", x=120, y=-20)
    except Exception as e:
        print("Falha ao anexar botão do Validador:", e)
