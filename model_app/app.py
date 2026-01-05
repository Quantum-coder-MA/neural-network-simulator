from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import time
import os
import base64
import io
from PIL import Image
# --- OPTIMISATION GPU (A mettre juste après les imports) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Configuration de l'application
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Variable globale pour garder le modèle en mémoire entre l'entraînement et la prédiction
model = None 

# --- 1. FONCTIONS UTILITAIRES (Helpers) ---

VISUAL_LIMIT = 16 # On ne montre que 16 neurones max par couche pour ne pas faire laguer le navigateur

def slice_data(data_array):
    """Coupe les matrices trop grandes pour la visualisation Web."""
    data_array = np.array(data_array)
    # Si c'est 1D (biais ou output de couche dense)
    if data_array.ndim == 1:
        return data_array[:VISUAL_LIMIT].tolist()
    # Si c'est 2D (poids Dense)
    elif data_array.ndim == 2:
        return data_array[:VISUAL_LIMIT, :VISUAL_LIMIT].tolist()
    # Si c'est 3D ou 4D (Conv2D), on aplatit et on coupe
    else:
        return data_array.flatten()[:VISUAL_LIMIT].tolist()

def prepare_image(image_data, target_size):
    """Convertit l'image base64 du web en array numpy pour le modèle."""
    if "base64," in image_data:
        image_data = image_data.split(",")[1]
    
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L') # 'L' = Convertir en Noir & Blanc (Grayscale)
    
    # On redimensionne selon la taille attendue
    img = img.resize((target_size, target_size))
    
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Reshape pour (1, 80, 80, 1) -> (Batch, Height, Width, Channels)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1) 
    
    return img_array

# --- 2. CHARGEMENT DES DONNÉES OPTIMISÉ ---
try:
    print("Chargement des données...")
    X = pickle.load(open("/home/nigga/engine/X.pickle", "rb"))
    y = pickle.load(open("/home/nigga/engine/y.pickle", "rb"))
    
    # On prend juste un échantillon pour le test (ex: 2000 images) pour éviter le crash RAM
    # Si ça marche, tu pourras augmenter ce chiffre petit à petit
    X = X[:5000] 
    y = y[:5000]
    
    X = X / 255.0
    y = np.array(y)
    
    # Vérification de la forme des données
    # Si X est (N, 50, 50), on le transforme en (N, 50, 50, 1) pour Keras
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
        
    print(f"Données chargées ! Shape: {X.shape}") # Vérifie que c'est bien (N, 50, 50, 1)
    
except Exception as e:
    print(f"ERREUR CHARGEMENT DONNÉES : {e}")
    # Fallback data
    X = np.random.rand(100, 50, 50, 1)
    y = np.random.randint(0, 2, 100)
# --- 3. CUSTOM CALLBACK AVANCÉ (Entraînement) ---
class SimulationCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        # On envoie les données tous les 5 batches pour la fluidité
        if batch % 5 == 0:
            layers_data = []
            
            # On parcourt les couches pour extraire les poids
            for layer in self.model.layers:
                # On vérifie si la couche a des poids (Conv et Dense en ont)
                if len(layer.get_weights()) > 0:
                    weights = layer.get_weights()[0]
                    # biases = layer.get_weights()[1] if len(layer.get_weights()) > 1 else []
                    
                    layers_data.append({
                        'name': layer.name,
                        'weights': slice_data(weights) # C'est ici qu'on extrait les matrices
                    })
            
            socketio.emit('training_step', {
                'batch': batch,
                'loss': float(logs.get('loss')),
                'accuracy': float(logs.get('accuracy')),
                'layers': layers_data # Envoi des poids pour l'animation des liens
            })
            socketio.sleep(0.01) # Petit break pour le socket

    def on_epoch_end(self, epoch, logs=None):
        # Fin de l'époque : mise à jour des courbes classiques
        socketio.emit('epoch_update', {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss')),
            'accuracy': float(logs.get('accuracy')),
            'val_loss': float(logs.get('val_loss')) if logs.get('val_loss') else 0,
            'val_accuracy': float(logs.get('val_accuracy')) if logs.get('val_accuracy') else 0
        })

# --- 4. ROUTES FLASK & SOCKETIO ---

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def handle_training(config_data):
    global model 
    print("Configuration reçue du UI :", config_data)
    
    # 1. Dimensions réelles
    real_height = X.shape[1] 
    real_width = X.shape[2]
    real_channels = X.shape[3] if len(X.shape) > 3 else 1
    
    print(f"Modèle pour : {real_height}x{real_width}x{real_channels}")

    learning_rate = float(config_data.get("learning_rate", 0.001))
    epochs = int(config_data.get('epochs', 10))
    conv_layers = config_data.get("conv_layers", [])
    dense_layers = config_data.get("dense_layers", [])
    
    model = Sequential()
    
    # --- CORRECTION MAJEURE ICI ---
    # Au lieu d'ajouter une couche Input séparée, on la définit dans la première couche.
    # C'est beaucoup plus stable pour la visualisation par la suite.
    
    # Cas 1: S'il y a des couches de convolution
    if len(conv_layers) > 0:
        first = conv_layers[0]
        # On met input_shape DANS le premier Conv2D
        model.add(Conv2D(int(first["filters"]), (int(first["kernel"]), int(first["kernel"])), 
                         input_shape=(real_height, real_width, real_channels), 
                         name="conv_0", padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        
        # Les autres couches conv
        for i, conv in enumerate(conv_layers[1:]):
            model.add(Conv2D(int(conv["filters"]), (int(conv["kernel"]), int(conv["kernel"])), 
                             name=f"conv_{i+1}", padding='same'))
            model.add(Activation("relu"))
            model.add(MaxPooling2D((2,2)))
            
        model.add(Flatten(name="flatten"))

    # Cas 2: Si pas de conv (juste Dense), on doit quand même gérer l'input
    else:
        model.add(Flatten(input_shape=(real_height, real_width, real_channels), name="flatten"))

    # Couches Dense
    for i, units in enumerate(dense_layers):
        model.add(Dense(int(units), name=f"dense_{i}"))
        model.add(Activation("relu"))

    model.add(Dense(1, activation="sigmoid", name="output"))

    # --- SECURITÉ SUPPLEMENTAIRE ---
    # On force la création du graphe maintenant
    model.build((None, real_height, real_width, real_channels))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    try:
        model.fit(
            X, y,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[SimulationCallback()],
            verbose=1
        )
        emit('training_complete', {'status': 'done'})
    except Exception as e:
        print(f"Erreur Entraînement: {e}")
        model = None 
        emit('training_error', {'error': str(e)})
@socketio.on('predict_sample')
def handle_prediction(data):
    global model
    
    if model is None:
        emit('prediction_result', {'error': 'ERREUR: Modèle non entraîné !'})
        return

    try:
        # --- CORRECTION ICI ---
        # On demande la forme d'entrée au modèle global, pas à la couche 0
        # model.input_shape renvoie (None, 80, 80, 1)
        if hasattr(model, 'input_shape'):
            target_size = model.input_shape[1]
        else:
            # Fallback pour certaines versions
            target_size = model.layers[0].input.shape[1]
            
        if target_size is None: target_size = 80 # Sécurité ultime
        
        print(f"Prédiction demandée. Taille image cible : {target_size}x{target_size}")
        
        img_data = data.get('image')
        processed_img = prepare_image(img_data, target_size)
        
        # 1. PRÉDICTION CLASSIQUE (On sécurise le résultat)
        prediction_score = model.predict(processed_img, verbose=0)
        
        # Gestion propre du résultat (scalar ou array)
        if isinstance(prediction_score, list):
            prediction_score = prediction_score[0]
            
        final_prob = float(prediction_score[0][0]) if prediction_score.ndim > 1 else float(prediction_score[0])
        
        network_state = []
        
        # 2. VISUALISATION (Avec protection anti-crash)
        try:
            # On utilise model.inputs et outputs pour recréer le graphe de visu
            # C'est la méthode la plus robuste
            layer_outputs = [layer.output for layer in model.layers]
            activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
            
            all_activations = activation_model.predict(processed_img, verbose=0)
            
            if not isinstance(all_activations, list):
                all_activations = [all_activations]

            # Input (L'image elle-même)
            network_state.append({
                'layer_name': 'Input',
                'activations': slice_data(processed_img.flatten())
            })

            # Couches cachées
            for i, val in enumerate(all_activations):
                layer_name = model.layers[i].name
                # On prend le premier élément du batch [0]
                network_state.append({
                    'layer_name': layer_name,
                    'activations': slice_data(val[0]) 
                })
                
        except Exception as e_visu:
            print(f"Info: Visualisation non disponible ({e_visu}), mais prédiction OK.")
            network_state = []

        # 3. ENVOI
        emit('prediction_simulation', {
            'network_state': network_state,
            'probability': final_prob
        })
        print(f"Prédiction envoyée : {final_prob}")
        
    except Exception as e:
        print(f"Erreur Générale Prédiction: {e}")
        import traceback
        traceback.print_exc() # Affiche l'erreur exacte dans le terminal pour débugger
        emit('prediction_result', {'error': str(e)})
if __name__ == '__main__':
    # use_reloader=False est souvent nécessaire avec TF+SocketIO pour éviter de charger le modèle 2 fois
    socketio.run(app, port=5000, debug=True, use_reloader=False)