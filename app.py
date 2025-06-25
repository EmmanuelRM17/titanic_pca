from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados
try:
    modelo_rf = joblib.load('modelo_random_forest.pkl')
    modelo_mlp = joblib.load('modelo_red_neuronal.pkl')
    app.logger.debug("Modelos cargados correctamente.")
except FileNotFoundError as e:
    app.logger.error(f"Error cargando modelos: {e}")
    modelo_rf = None
    modelo_mlp = None

# Mapeos para variables categóricas (según análisis del dataset)
CUT_MAPPING = {
    'Fair': 0,
    'Good': 1, 
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}

COLOR_MAPPING = {
    'J': 0,
    'I': 1,
    'H': 2,
    'G': 3,
    'F': 4,
    'E': 5,
    'D': 6
}

CLARITY_MAPPING = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que los modelos estén cargados
    if modelo_rf is None or modelo_mlp is None:
        return jsonify({'error': 'Modelos no disponibles'}), 500
    
    try:
        # Obtener datos del formulario
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        modelo_usado = request.form.get('modelo', 'rf')
        
        # Validar que las categorías existan en los mapeos
        if cut not in CUT_MAPPING:
            raise ValueError(f"Tipo de corte '{cut}' no válido")
        if color not in COLOR_MAPPING:
            raise ValueError(f"Color '{color}' no válido")
        if clarity not in CLARITY_MAPPING:
            raise ValueError(f"Claridad '{clarity}' no válida")
        
        # Codificar variables categóricas
        cut_encoded = CUT_MAPPING[cut]
        color_encoded = COLOR_MAPPING[color]
        clarity_encoded = CLARITY_MAPPING[clarity]
        
        # Calcular volumen
        volume = x * y * z
        
        # Crear DataFrame con las características en el orden correcto
        # Ajusta este orden según cómo entrenaste tu modelo
        feature_names = [
            'carat', 'depth', 'table', 'x', 'y', 'z', 
            'cut_encoded', 'color_encoded', 'clarity_encoded', 'volume'
        ]
        
        data_df = pd.DataFrame([[
            carat, depth, table, x, y, z,
            cut_encoded, color_encoded, clarity_encoded, volume
        ]], columns=feature_names)
        
        app.logger.debug(f"Datos procesados: {data_df.iloc[0].to_dict()}")
        
        # Realizar predicción
        if modelo_usado == 'mlp':
            prediction = modelo_mlp.predict(data_df)
            app.logger.debug("Usando modelo Red Neuronal")
        else:
            prediction = modelo_rf.predict(data_df)
            app.logger.debug("Usando modelo Random Forest")
        
        precio_estimado = float(prediction[0])
        app.logger.debug(f"Predicción realizada: ${precio_estimado:.2f}")
        
        return jsonify({
            'precio_estimado': precio_estimado,
            'modelo_usado': 'Red Neuronal' if modelo_usado == 'mlp' else 'Random Forest'
        })
        
    except ValueError as e:
        app.logger.error(f"Error de validación: {str(e)}")
        return jsonify({'error': f'Datos inválidos: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)