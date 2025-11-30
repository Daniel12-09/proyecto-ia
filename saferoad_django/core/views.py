from django.shortcuts import render
from django.shortcuts import render
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. ENTRENAMOS LA IA AL INICIAR ---
# Datos simulados (Clima, Tráfico, Velocidad -> Accidente)
datos = {
    'Lluvia': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    'Trafico': [1, 8, 2, 9, 1, 5, 3, 2, 9, 7],
    'Velocidad': [60,80,50,90,40,70,60,55,20,85],
    'Accidente': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(datos)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(df[['Lluvia', 'Trafico', 'Velocidad']], df['Accidente'])

def home(request):
    resultado = None
    # Valores iniciales
    lluvia = 0
    trafico = 5
    velocidad = 60

    if request.method == 'POST':
        # Recibir datos del formulario
        lluvia = int(request.POST.get('lluvia'))
        trafico = int(request.POST.get('trafico'))
        velocidad = int(request.POST.get('velocidad'))
        
        # IA Predice
        prediccion = modelo.predict([[lluvia, trafico, velocidad]])[0]
        probabilidad = modelo.predict_proba([[lluvia, trafico, velocidad]])[0][1]
        
        resultado = {
            'riesgo': int(prediccion),
            'probabilidad': round(probabilidad * 100, 1),
            'mensaje': "ALTO RIESGO 🔴" if prediccion == 1 else "CONDICIONES SEGURAS 🟢"
        }

    return render(request, 'home.html', {
        'resultado': resultado,
        'lluvia': lluvia,
        'trafico': trafico,
        'velocidad': velocidad
    })