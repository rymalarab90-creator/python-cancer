# %% md
# Premiére partie
# %%
# les biblithéques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# chargement de données
data = pd.read_csv("patients_cancer_poumon.csv")

# exploration de données
print("Affichage de la taille de données:", data.shape)

print("Affichage des 5 premiéres lignes:", data.head())

print("Identifications des variables;", data.columns.tolist)
print("Les types des données;", data.dtypes)
print("L'age moyen des fumeurs est:", data['age'].mean())
print('L\'age minimum des fumeurs est:', data['age'].min())
print('L\'age maximum des fumeurs est:', data['age'].max())
print('Le nombre de personnes par classe de risque:', data['risque_malignite'].value_counts())
print("Le nombre de patients par sexe:", data['sexe_masculin'].value_counts())
print("Le nombre de doublons:", data.duplicated().sum())

# la vérification des valeurs nulles
print("Affichage des valeurs nulles;", data.isnull().sum())

# Des statistiques
print(data.describe())

# Les corrélations

# Relation entre taille du nodule et risque de malignité
plt.figure(figsize=(8, 6))
plt.scatter(data['taille_nodule_px'], data['risque_malignite'], alpha=0.6, color='blue')
plt.title("Relation entre taille du nodule et risque de malignité")
plt.xlabel("Taille du nodule (px)")
plt.ylabel("Risque de malignité")
plt.grid(True)
plt.show()

# Age vs Tabagisme coloré par Risque de Malignité
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    data['age'],
    data['tabagisme_paquets_annee'],
    c=data['risque_malignite'],  # couleur selon le risque
    cmap='viridis',  # palette de couleurs
    alpha=0.7
)

plt.xlabel("Âge")
plt.ylabel("Tabagisme (paquets/année)")
plt.title("Âge vs Tabagisme coloré par Risque de Malignité")
plt.colorbar(scatter, label="Risque de Malignité")  # barre de couleur

plt.grid(True)
plt.show()

# Relation entre antécédent familial et risque de malignité"
plt.figure(figsize=(7, 5))

# Scatter plot avec un léger jitter pour mieux voir les points (utile si valeurs 0 ou 1)
x_jitter = data['antecedent_familial'] + np.random.normal(0, 0.05, size=len(data))

plt.scatter(x_jitter, data['risque_malignite'], alpha=0.6, color='red')
plt.xlabel("Antécédent familial (0 = non, 1 = oui)")
plt.ylabel("Risque de malignité")
plt.title("Relation entre antécédent familial et risque de malignité")
plt.grid(True)
plt.show()


# Visualisation de l'Imagerie (JSRT Subset)

def plot_samples_from_parent_folder(parent_folder, n=3):
    # Liste des sous-dossiers (classes)
    classes = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    classes.sort()  # ordre fixe

    # Créer la grille de sous-plots
    fig, axes = plt.subplots(len(classes), n, figsize=(4 * n, 4 * len(classes)))

    # Si axes n'est pas un tableau 2D
    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=0)
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, cls in enumerate(classes):
        folder_path = os.path.join(parent_folder, cls)

        # Récupérer toutes les images
        img_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_files) == 0:
            print(f"Aucune image trouvée dans {folder_path}")
            continue

        # Mélanger et prendre les n premières
        np.random.shuffle(img_files)
        samples = img_files[:n]

        for j, img_name in enumerate(samples):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert('L')
            except Exception as e:
                print(f"Impossible d'ouvrir {img_path}: {e}")
                continue

            # Affichage
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{cls}\n{img_name}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Radios thoraciques représentatives par classe (JSRT)", fontsize=16, y=1.05)
    plt.show()


# Utilisation
parent_folder = r'C:\Users\celia\Esic TP noté ILDE\TP images'  # dossier parent contenant sain/benin/malin
plot_samples_from_parent_folder(parent_folder, n=3)

# %% md
# Deuxiéme partie
# %% raw
Le
premier
modéle
# %%
y = data['risque_malignite']  # la colonne qui contient les classes (0,1,2)
X = data.drop(columns=['risque_malignite', 'patient_id', 'classe_jsrt_source', 'diagnostic_source',
                       'image_path'])  # toutes les colonnes sauf le label

# train_ttest_split

# Supposons que X = tes features (par ex. LBP ou autres)
# et y = tes labels numériques (0,1,2)

# Diviser les données : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% pour le test
    random_state=42,  # pour reproductibilité
    stratify=y  # pour garder la proportion des classes
)

print(f"Nombre d'échantillons train : {len(X_train)}")
print(f"Nombre d'échantillons test : {len(X_test)}")

# 1er modéle


# --- Créer le modèle (multinomial = multi-classes) ---
model_lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# --- Entraînement ---
model_lr.fit(X_train, y_train)

# --- Prédiction ---
y_pred_lr = model_lr.predict(X_test)

# --- Évaluation ---
print(classification_report(y_test, y_pred_lr))

# --- Sauvegarde des prédictions dans un fichier CSV ---
results_df = pd.DataFrame({
    'index': X_test.index,  # ou 'patient_id' si disponible
    'y_true': y_test,
    'y_pred': y_pred_lr
})

results_df.to_csv("predictions_logistic_regression.csv", index=False)
print("Les prédictions ont été sauvegardées dans 'predictions_logistic_regression.csv'")
# %% raw
Le
2
eme
modéle
# %% raw
1
er
version: Classification
d
'images
# %%
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Chemin vers les dossiers ---
base_dir = r'C:\Users\celia\Esic TP noté ILDE\TP images'
sain_dir = os.path.join(base_dir, 'sain')
benin_dir = os.path.join(base_dir, 'benin')
malin_dir = os.path.join(base_dir, 'malin')

# --- Créer des DataFrames avec chemins et labels ---
data_list = []

# Classe 0 : sain
for img_name in os.listdir(sain_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        data_list.append([os.path.join(sain_dir, img_name), 0])

# Classe 1 : benin + malin
for folder in [benin_dir, malin_dir]:
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            data_list.append([os.path.join(folder, img_name), 1])

df = pd.DataFrame(data_list, columns=['image_path', 'label'])

# --- Convertir les labels en chaînes de caractères pour Keras ---
df['label'] = df['label'].astype(str)

# --- Split train / test stratifié ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# --- DataGenerator simple (juste rescale) ---
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# --- Générateurs ---
img_size = (128, 128)
batch_size = 16

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=img_size,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label',
    target_size=img_size,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False
)

# --- Définition du CNN ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # sortie binaire
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# --- Entraînement ---
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20
)

# --- Prédictions sur test set ---
test_generator.reset()
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# --- Évaluation ---
y_true = test_df['label'].astype(int).values  # convert en int pour classification_report
print(classification_report(y_true, y_pred))

# --- Sauvegarder les prédictions ---
results_df = pd.DataFrame({
    'image_path': test_df['image_path'],
    'y_true': y_true,
    'y_pred': y_pred
})
results_df.to_csv("predictions_cnn.csv", index=False)
print("Les prédictions CNN ont été sauvegardées dans 'predictions_cnn.csv'")
# %% raw
2
éme
version
# %%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Chargement des fichiers de prédictions obtenue dans le modéle 1 (régression logistique) et dans le modéle 2 (version 1: CNN)
pred_cnn = pd.read_csv("predictions_cnn.csv")
pred_reglog = pd.read_csv("predictions_logistic_regression.csv")  # colonnes: index,y_true,y_pred

# Harmonisation des labels : remplacer 2 par 1 dans les prédictions de la régression logistique
pred_reglog['y_pred'] = pred_reglog['y_pred'].replace(2, 1)
pred_reglog['y_true'] = pred_reglog['y_true'].replace(2, 1)

# Création d'une DataFrame avec les  prédictions des deux modéles

X = pd.DataFrame({
    'cnn': pred_cnn['y_pred'],
    'reglog': pred_reglog['y_pred']
})
y = pred_cnn['y_true']

print(X)
print("les vrai val", y)
# Régression logistique finale
model = LogisticRegression()
model.fit(X, y)
y_pred_final = model.predict(X)

# Évaluation de modéle
print("=== Rapport classification final ===")
print(classification_report(y, y_pred_final))

# Sauvegarde des résultats modéle 2 (version 2)
results = X.copy()
results['y_true'] = y
results['y_pred_final'] = y_pred_final
results.to_csv("predictions_modéle 2 (V2).csv", index=False)
print("Les prédictions combinées ont été sauvegardées dans 'predictions_modéle 2 (V2).csv'")
# %%


# On va afficher les N premières images avec leur prédiction finale
N = 10  # nombre d'images à afficher
subset = results.head(N)

plt.figure(figsize=(15, 5))

for i, row in enumerate(subset.itertuples()):
    img_path = row.Index if 'image_path' in subset.columns else pred_cnn.loc[i, 'image_path']
    img = Image.open(img_path).convert('L')
    plt.subplot(2, N // 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Vrai: {row.y_true}\nPred: {row.y_pred_final}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
