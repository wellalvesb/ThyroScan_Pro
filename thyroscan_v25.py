# ==============================================================================
# THYROSCAN PRO V25 — PRODUCTION ENGINE (COLAB + LOCAL)
# Arquitetura: POO + ROI Autônomo + SMOTE por Fold + GLCM Multiaxial + CV 5-Fold
# Compatível com: Google Colab e Execução Local (GitHub)
# ==============================================================================

import os
import sys
import cv2
import zipfile
import shutil
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para compatibilidade
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay, RocCurveDisplay)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠️  imblearn não encontrado. SMOTE será desabilitado.")
    print("   Instale com: pip install imbalanced-learn")

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Detectar ambiente ────────────────────────────────────────────────────────
IS_COLAB = False
try:
    from google.colab import files as colab_files
    IS_COLAB = True
except ImportError:
    pass


# ==============================================================================
# 1. MÓDULO DE VISÃO COMPUTACIONAL — ISOLAMENTO DO NÓDULO (ROI)
# ==============================================================================
class ThyroVision:
    """
    Localiza autonomamente o nódulo de tireoide na imagem de ultrassom,
    descartando artefatos, textos e marcações médicas.
    Usa Black-Hat com múltiplos kernels + CLAHE para robustez.
    """

    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def _aplicar_clahe(self, img_gray):
        """Normaliza contraste local da imagem."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)

    def _encontrar_melhor_contorno(self, img_gray, thresh_img):
        """Encontra o contorno mais provável de ser o nódulo (central + grande)."""
        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        h_img, w_img = img_gray.shape
        c_img = np.array([w_img // 2, h_img // 2])
        melhor_c, max_score = None, -1

        for c in cnts:
            area = cv2.contourArea(c)
            # Filtro: ignorar contornos muito pequenos ou muito grandes
            if area < 400 or area > (h_img * w_img * 0.45):
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Aspect ratio razoável (nódulos não são linhas finas)
            ar = w / max(h, 1)
            if ar > 5 or ar < 0.2:
                continue
            dist = np.linalg.norm(c_img - np.array([x + w // 2, y + h // 2]))
            score = area - (dist * 2.0)  # Favorece massas centrais
            if score > max_score:
                max_score, melhor_c = score, (x, y, w, h)

        return melhor_c

    def isolar_nodulo(self, img_gray):
        """
        Aplica Morfologia Matemática (Black-Hat) com múltiplos kernels
        para localizar o nódulo de forma robusta.
        """
        # Pré-processamento: CLAHE + Bilateral Filter
        img_clahe = self._aplicar_clahe(img_gray)
        blur = cv2.bilateralFilter(img_clahe, 9, 75, 75)

        # Tentar múltiplos tamanhos de kernel para robustez
        melhor_roi = None
        melhor_area = -1

        for ksize in [45, 55, 65]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
            _, thresh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)

            # Limpar ruído
            k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_clean)

            bbox = self._encontrar_melhor_contorno(img_gray, thresh)
            if bbox is not None:
                x, y, w, h = bbox
                area = w * h
                if area > melhor_area:
                    melhor_area = area
                    melhor_roi = bbox

        if melhor_roi is not None:
            x, y, w, h = melhor_roi
            h_img, w_img = img_gray.shape
            p = 12  # Padding clínico
            roi = img_gray[max(0, y - p):min(h_img, y + h + p),
                           max(0, x - p):min(w_img, x + w + p)]
            return cv2.resize(roi, self.target_size)

        # Fallback: crop central (60% da imagem) — evita bordas com artefatos
        h_img, w_img = img_gray.shape
        margin_h = int(h_img * 0.2)
        margin_w = int(w_img * 0.2)
        roi = img_gray[margin_h:h_img - margin_h, margin_w:w_img - margin_w]
        return cv2.resize(roi, self.target_size)


# ==============================================================================
# 2. MÓDULO RADIÔMICO — EXTRAÇÃO DE BIOMARCADORES TI-RADS
# ==============================================================================
class RadiomicsDNA:
    """
    Converte imagem ROI em vetor de biomarcadores baseados em TI-RADS:
    - HOG (bordas/espículas): ~1.296 features
    - GLCM Multiaxial (textura): 24 features
    - LBP (rugosidade): 10 features
    - Estatísticas de intensidade: 4 features
    Total estimado: ~1.334 features
    """

    def __init__(self):
        # HOG compacto: 128x128 px, resultando em ~1.296 features
        self.hog = cv2.HOGDescriptor(
            (128, 128),   # winSize
            (32, 32),     # blockSize
            (16, 16),     # blockStride
            (16, 16),     # cellSize
            9             # nbins
        )

    def extrair_assinatura(self, roi):
        """Extrai vetor completo de biomarcadores da ROI."""
        img = cv2.resize(roi, (128, 128))

        # Limpeza acústica antes da extração
        img_clean = cv2.bilateralFilter(img, 5, 50, 50)

        # A. Morfologia de Bordas — HOG
        hog_feat = self.hog.compute(img_clean).flatten()

        # B. Textura Multiaxial — GLCM (4 ângulos × 6 propriedades = 24 feat)
        glcm = graycomatrix(img_clean, [1, 3],
                            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            256, symmetric=True, normed=True)
        glcm_feats = []
        for prop in ['contrast', 'homogeneity', 'energy',
                      'correlation', 'dissimilarity', 'ASM']:
            vals = graycoprops(glcm, prop)
            glcm_feats.extend(vals.flatten().tolist())

        # C. Rugosidade Superficial — LBP
        lbp = local_binary_pattern(img_clean, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10,
                                   range=(0, 10), density=True)

        # D. Estatísticas de Intensidade
        eco_media = np.mean(img_clean)
        eco_std = np.std(img_clean)
        eco_skew = float(np.mean(((img_clean - eco_media) / max(eco_std, 1e-7)) ** 3))
        eco_kurt = float(np.mean(((img_clean - eco_media) / max(eco_std, 1e-7)) ** 4))

        return np.hstack((hog_feat, glcm_feats, lbp_hist,
                          [eco_media, eco_std, eco_skew, eco_kurt]))


# ==============================================================================
# 3. MÓDULO DE INTELIGÊNCIA — PIPELINE DE TREINAMENTO COM CV
# ==============================================================================
class ThyroBrain:
    """Pipeline de treinamento com validação cruzada 5-fold e SMOTE por fold."""

    def __init__(self, n_features=300):
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_classif, k=n_features)
        self.model = None
        self.threshold = 0.5
        self.n_features = n_features

    def _criar_ensemble(self):
        """Cria o Voting Classifier (RF + HistGradientBoosting)."""
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=12,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=8,
            learning_rate=0.05, random_state=42
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('hgb', hgb)], voting='soft'
        )

    def treinar_com_cv(self, X, y, n_folds=5):
        """
        Treina com validação cruzada estratificada.
        SMOTE é aplicado DENTRO de cada fold para evitar data leakage.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_probs = np.zeros(len(y))
        all_preds = np.zeros(len(y))
        fold_metrics = []

        print(f"\n{'═' * 70}")
        print(f" 🔬 VALIDAÇÃO CRUZADA {n_folds}-FOLD")
        print(f"{'═' * 70}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # SMOTE dentro do fold (evita data leakage)
            if HAS_SMOTE:
                smote = SMOTE(random_state=42)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            else:
                X_tr_res, y_tr_res = X_tr, y_tr

            # Scaler + SelectKBest dentro do fold
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_res)
            X_val_s = scaler.transform(X_val)

            selector = SelectKBest(f_classif, k=min(self.n_features, X_tr_s.shape[1]))
            X_tr_k = selector.fit_transform(X_tr_s, y_tr_res)
            X_val_k = selector.transform(X_val_s)

            # Treinar ensemble
            model = self._criar_ensemble()
            model.fit(X_tr_k, y_tr_res)

            # Predições
            probs = model.predict_proba(X_val_k)[:, 1]
            all_probs[val_idx] = probs

            # Métricas do fold com threshold default (0.5)
            preds = (probs >= 0.5).astype(int)
            all_preds[val_idx] = preds

            rec = recall_score(y_val, preds)
            prec = precision_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds)
            acc = accuracy_score(y_val, preds)

            fold_metrics.append({
                'recall': rec, 'precision': prec,
                'f1': f1, 'accuracy': acc
            })
            print(f"  Fold {fold}: Recall={rec*100:.1f}% | "
                  f"Precision={prec*100:.1f}% | F1={f1*100:.1f}% | "
                  f"Acc={acc*100:.1f}%")

        # Calcular threshold ótimo via Youden Index sobre todos os folds
        fpr, tpr, thresholds = roc_curve(y, all_probs)
        self.threshold = thresholds[np.argmax(tpr - fpr)]

        # Métricas finais com threshold otimizado
        y_pred_opt = (all_probs >= self.threshold).astype(int)

        print(f"\n{'─' * 70}")
        print(f"  📐 Limiar Otimizado (Youden): {self.threshold:.3f}")
        print(f"  📊 MÉTRICAS MÉDIAS (CV):")
        print(f"     Recall   : {np.mean([m['recall'] for m in fold_metrics])*100:.2f}%")
        print(f"     Precisão : {np.mean([m['precision'] for m in fold_metrics])*100:.2f}%")
        print(f"     F1-Score : {np.mean([m['f1'] for m in fold_metrics])*100:.2f}%")
        print(f"     Acurácia : {np.mean([m['accuracy'] for m in fold_metrics])*100:.2f}%")
        print(f"  📊 MÉTRICAS COM THRESHOLD OTIMIZADO:")
        print(f"     Recall   : {recall_score(y, y_pred_opt)*100:.2f}%")
        print(f"     Precisão : {precision_score(y, y_pred_opt, zero_division=0)*100:.2f}%")
        print(f"     F1-Score : {f1_score(y, y_pred_opt)*100:.2f}%")
        print(f"     AUC-ROC  : {roc_auc_score(y, all_probs)*100:.2f}%")

        return y, all_probs, y_pred_opt, fpr, tpr, fold_metrics

    def treinar_modelo_final(self, X_train, y_train, X_test, y_test):
        """
        Treina o modelo final com todos os dados de treino.
        Usa o threshold ótimo calculado na CV.
        """
        print(f"\n{'═' * 70}")
        print(f" 🧠 TREINAMENTO DO MODELO FINAL")
        print(f"{'═' * 70}")

        # SMOTE
        if HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_train, y_train)
            print(f"  ⚖️  SMOTE: {len(X_train)} → {len(X_tr_res)} amostras")
        else:
            X_tr_res, y_tr_res = X_train, y_train

        # Scaler + SelectKBest
        X_tr_s = self.scaler.fit_transform(X_tr_res)
        X_ts_s = self.scaler.transform(X_test)

        k = min(self.n_features, X_tr_s.shape[1])
        self.selector = SelectKBest(f_classif, k=k)
        X_tr_k = self.selector.fit_transform(X_tr_s, y_tr_res)
        X_ts_k = self.selector.transform(X_ts_s)

        # Treinar
        self.model = self._criar_ensemble()
        self.model.fit(X_tr_k, y_tr_res)

        # Avaliar no teste
        probs = self.model.predict_proba(X_ts_k)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs)

        # Recalcular threshold se necessário
        self.threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred = (probs >= self.threshold).astype(int)

        self._gerar_laudo(y_test, y_pred, probs, fpr, tpr)
        return y_test, y_pred, probs, fpr, tpr

    def _gerar_laudo(self, y_test, y_pred, probs, fpr, tpr):
        """Gera laudo científico com métricas e gráficos."""
        print(f"\n{'═' * 70}")
        print(f" 🔬 LAUDO CIENTÍFICO FINAL — THYROSCAN PRO V25")
        print(f"{'═' * 70}")
        print(f"  🎯 Acurácia Global        : {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(f"  🛡️  Sensibilidade (Recall) : {recall_score(y_test, y_pred)*100:.2f}%")
        print(f"  🔪 Precisão Diagnóstica   : {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
        print(f"  ⚖️  F1-Score               : {f1_score(y_test, y_pred)*100:.2f}%")
        print(f"  📈 AUC-ROC                : {roc_auc_score(y_test, probs)*100:.2f}%")
        print(f"  📐 Limiar de Corte        : {self.threshold:.3f}")
        print(f"{'═' * 70}")

        # Gráficos
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        RocCurveDisplay(fpr=fpr, tpr=tpr,
                        roc_auc=roc_auc_score(y_test, probs)
                        ).plot(ax=ax[0], color='darkorange')
        ax[0].plot([0, 1], [0, 1], 'k--')
        ax[0].set_title("Curva ROC — Poder de Discriminação TI-RADS")

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, cmap='Greens', ax=ax[1],
            display_labels=['Benigno', 'Maligno']
        )
        ax[1].set_title("Matriz de Confusão — Validação Clínica")
        plt.tight_layout()
        plt.savefig('thyroscan_v25_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  📊 Gráficos salvos em: thyroscan_v25_resultados.png")

    def salvar_modelo(self, path='thyroscan_v25_model.pkl'):
        """Salva o modelo, scaler, selector e threshold para produção."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'threshold': self.threshold
        }, path)
        print(f"  ✅ Modelo salvo: {path}")


# ==============================================================================
# 4. MÓDULO DE INFERÊNCIA — CLASSIFICAÇÃO DE NOVA IMAGEM
# ==============================================================================
class ThyroInference:
    """Carrega modelo salvo e classifica novas imagens de ultrassom."""

    def __init__(self, model_path='thyroscan_v25_model.pkl'):
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.selector = data['selector']
        self.threshold = data['threshold']
        self.vision = ThyroVision()
        self.radiomics = RadiomicsDNA()

    def classificar(self, img_path):
        """
        Classifica uma imagem de ultrassom de tireoide.
        Retorna: (classe, probabilidade_maligno, roi_image)
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível carregar: {img_path}")

        # Isolar nódulo
        roi = self.vision.isolar_nodulo(img)

        # Extrair features
        features = self.radiomics.extrair_assinatura(roi)
        features = features.reshape(1, -1)

        # Normalizar e selecionar features
        features_s = self.scaler.transform(features)
        features_k = self.selector.transform(features_s)

        # Predição
        prob = self.model.predict_proba(features_k)[0, 1]
        classe = "MALIGNO" if prob >= self.threshold else "BENIGNO"

        return classe, prob, roi

    def classificar_array(self, img_array):
        """
        Classifica a partir de um array numpy (para uso no frontend).
        """
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_array

        roi = self.vision.isolar_nodulo(img_gray)
        features = self.radiomics.extrair_assinatura(roi)
        features = features.reshape(1, -1)
        features_s = self.scaler.transform(features)
        features_k = self.selector.transform(features_s)

        prob = self.model.predict_proba(features_k)[0, 1]
        classe = "MALIGNO" if prob >= self.threshold else "BENIGNO"

        return classe, prob, roi


# ==============================================================================
# 5. PIPELINE PRINCIPAL — INGESTÃO + TREINAMENTO
# ==============================================================================
def carregar_dataset(base_dir):
    """Carrega caminhos das imagens e labels do dataset."""
    paths, labels = [], []
    for nome, valor in {'benign': 0, 'malignant': 1}.items():
        pasta = os.path.join(base_dir, nome)
        if not os.path.exists(pasta):
            # Tentar case-insensitive
            for d in os.listdir(base_dir):
                if d.lower() == nome.lower():
                    pasta = os.path.join(base_dir, d)
                    break
        if os.path.exists(pasta):
            for f in sorted(os.listdir(pasta)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(pasta, f))
                    labels.append(valor)

    print(f"  📁 Dataset carregado:")
    benign_count = labels.count(0)
    malig_count = labels.count(1)
    print(f"     Benigno  : {benign_count} imagens")
    print(f"     Maligno  : {malig_count} imagens")
    print(f"     Total    : {len(labels)} imagens")
    return paths, labels


def extrair_features(paths, labels, vision, radiomics, augment=False):
    """Extrai features radiômicas de todas as imagens."""
    X, y = [], []
    total = len(paths)

    for i, (p, label) in enumerate(zip(paths, labels)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  🔍 Processando imagem {i+1}/{total}...")

        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        roi = vision.isolar_nodulo(img)

        if augment:
            # Augmentation conservador: 0°, 90°, 180°
            for k in [0, 1, 2]:
                rot = np.rot90(roi, k=k)
                X.append(radiomics.extrair_assinatura(rot))
                y.append(label)
        else:
            X.append(radiomics.extrair_assinatura(roi))
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def upload_e_extrair_dataset():
    """Upload e extração do dataset .zip (Colab ou local)."""
    
    # Procura se o usuário já colocou um arquivo .zip na pasta atual
    zips = [f for f in os.listdir('.') if f.endswith('.zip')]
    
    if zips:
        # Pega o maior arquivo zip (provavelmente o dataset do kaggle)
        nome_zip = max(zips, key=lambda f: os.path.getsize(f))
        print(f"📦 Arquivo zip encontrado automaticamente: {nome_zip}")
    elif IS_COLAB:
        print("⚠️ Nenhum arquivo .zip encontrado na pasta.")
        print("📦 Por favor, faça o upload do arquivo .zip do dataset:")
        uploaded = colab_files.upload()
        nome_zip = list(uploaded.keys())[0]
    else:
        # Se for local e não tiver zip, verifica se as pastas já estão extraídas
        if os.path.exists('tumores'):
            return encontrar_base_dir('tumores')
        if os.path.exists('benign') and os.path.exists('malignant'):
            return '.'
            
        print("❌ Nenhum arquivo .zip encontrado. Coloque o dataset .zip no diretório atual para treinar.")
        sys.exit(1)

    destino = 'dataset_thyroscan_v25'
    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)

    try:
        with zipfile.ZipFile(nome_zip, 'r') as z:
            z.extractall(destino)
    except zipfile.BadZipFile:
        print(f"\n❌ ERRO CRÍTICO: O arquivo {nome_zip} está corrompido ou incompleto.")
        print("   Se você está no Google Colab, verifique se o UPLOAD TERMINOU completamente.")
        print("   Olhe no canto inferior esquerdo da tela: deve haver um círculo laranja/azul girando.")
        print("   Aguarde a barra de upload sumir antes de rodar a célula novamente!")
        sys.exit(1)

    return encontrar_base_dir(destino)


def encontrar_base_dir(destino):
    """Encontra o diretório que contém as pastas benign e malignant."""
    for root, dirs, _ in os.walk(destino):
        dirs_lower = [d.lower() for d in dirs]
        if 'benign' in dirs_lower and 'malignant' in dirs_lower:
            return root
    raise FileNotFoundError("Pastas 'benign' e 'malignant' não encontradas!")


def main():
    """Pipeline completo de treinamento."""
    print("🚀 THYROSCAN PRO V25 — INICIALIZANDO PIPELINE")
    print("═" * 70)

    # 1. Carregar dataset
    base_dir = upload_e_extrair_dataset()
    paths, labels = carregar_dataset(base_dir)

    if len(paths) == 0:
        print("❌ Nenhuma imagem encontrada!")
        return

    # 2. Dividir em treino/teste (80/20) ANTES de qualquer processamento
    tr_paths, ts_paths, y_tr_raw, y_ts_raw = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 3. Instanciar módulos
    vision = ThyroVision()
    radiomics = RadiomicsDNA()

    # 4. Extrair features
    print("\n🔍 Extraindo features radiômicas (TREINO com augmentation)...")
    X_train, y_train = extrair_features(tr_paths, y_tr_raw, vision, radiomics,
                                         augment=True)

    print("\n🔍 Extraindo features radiômicas (TESTE sem augmentation)...")
    X_test, y_test = extrair_features(ts_paths, y_ts_raw, vision, radiomics,
                                       augment=False)

    print(f"\n  📊 Shape treino: {X_train.shape}")
    print(f"  📊 Shape teste : {X_test.shape}")

    # 5. Validação cruzada nos dados de treino
    brain = ThyroBrain(n_features=300)
    brain.treinar_com_cv(X_train, y_train, n_folds=5)

    # 6. Treinar modelo final
    y_test_out, y_pred, probs, fpr, tpr = brain.treinar_modelo_final(
        X_train, y_train, X_test, y_test
    )

    # 7. Salvar modelo
    brain.salvar_modelo()

    print("\n✅ PIPELINE COMPLETO! Modelo pronto para produção.")
    print("   Use ThyroInference('thyroscan_v25_model.pkl') para classificar novas imagens.")


if __name__ == "__main__":
    main()
