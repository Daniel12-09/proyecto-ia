from django.shortcuts import render
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier

# Importamos TensorFlow para la LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# ==========================================
# 🧠 ZONA DE ENTRENAMIENTO (SE EJECUTA AL INICIO)
# ==========================================
print("⏳ Iniciando entrenamiento de modelos...")

# --- 1. ENTRENAR RANDOM FOREST (RF) ---
datos_rf = {
    'Lluvia': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    'Trafico': [1, 8, 2, 9, 1, 5, 3, 2, 9, 7],
    'Velocidad': [60,80,50,90,40,70,60,55,20,85],
    'Accidente': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
}
df_rf = pd.DataFrame(datos_rf)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(df_rf[['Lluvia', 'Trafico', 'Velocidad']], df_rf['Accidente'])
print("✅ Random Forest listo.")

# --- 2. ENTRENAR LSTM (Red Neuronal) ---
# Generamos datos secuenciales para LSTM (simulamos 3 pasos de tiempo)
X_train_lstm = []
y_train_lstm = []

for i in range(100):
    if i % 2 == 0: # Caso Seguro
        secuencia = [[0, 1, 60], [0, 2, 60], [0, 2, 60]]
        etiqueta = 0
    else: # Caso Peligroso
        secuencia = [[0, 5, 50], [1, 7, 40], [1, 9, 20]]
        etiqueta = 1
    X_train_lstm.append(secuencia)
    y_train_lstm.append(etiqueta)

X_train_lstm = np.array(X_train_lstm)
y_train_lstm = np.array(y_train_lstm)

modelo_lstm = Sequential()
modelo_lstm.add(LSTM(units=50, activation='relu', input_shape=(3, 3)))
modelo_lstm.add(Dense(1, activation='sigmoid'))
modelo_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, verbose=0)
print("✅ LSTM lista.")

# ==========================================
# 🎮 CONTROLADOR DE LA VISTA
# ==========================================
def home(request):
    resultado = None
    lluvia = 0; trafico = 5; velocidad = 60
    tipo_modelo = "rf" # Por defecto

    if request.method == 'POST':
        lluvia = int(request.POST.get('lluvia'))
        trafico = int(request.POST.get('trafico'))
        velocidad = int(request.POST.get('velocidad'))
        tipo_modelo = request.POST.get('modelo_seleccionado') # Recibimos la elección

        riesgo = 0
        probabilidad = 0.0
        mensaje_modelo = ""

        # --- OPCIÓN A: RANDOM FOREST ---
        if tipo_modelo == 'rf':
            pred = modelo_rf.predict([[lluvia, trafico, velocidad]])[0]
            prob = modelo_rf.predict_proba([[lluvia, trafico, velocidad]])[0][1]
            
            riesgo = int(pred)
            probabilidad = round(prob * 100, 1)
            mensaje_modelo = "Análisis Estadístico (RF)"

        # --- OPCIÓN B: LSTM (Deep Learning) ---
        elif tipo_modelo == 'lstm':
            # Simulamos la historia temporal para la LSTM
            paso_t_2 = [lluvia, max(0, trafico-2), min(120, velocidad+10)]
            paso_t_1 = [lluvia, max(0, trafico-1), min(120, velocidad+5)]
            paso_actual = [lluvia, trafico, velocidad]
            
            secuencia = np.array([[paso_t_2, paso_t_1, paso_actual]])
            
            pred_raw = modelo_lstm.predict(secuencia)[0][0]
            riesgo = 1 if pred_raw > 0.5 else 0
            probabilidad = round(float(pred_raw) * 100, 1)
            mensaje_modelo = "Análisis Neuronal Temporal (LSTM)"

        # Resultado final
        resultado = {
            'riesgo': riesgo,
            'probabilidad': probabilidad,
            'origen': mensaje_modelo,
            'mensaje': "ALTO RIESGO 🔴" if riesgo == 1 else "CONDICIONES SEGURAS 🟢"
        }

    return render(request, 'home.html', {
        'resultado': resultado,
        'lluvia': lluvia,
        'trafico': trafico,
        'velocidad': velocidad,
        'modelo_actual': tipo_modelo
    })