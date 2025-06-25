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

# CARACTERÍSTICAS EXACTAS QUE ESPERAN LOS MODELOS (según debug)
REQUIRED_FEATURES = ['volume', 'clarity_encoded', 'carat', 'color_encoded', 'depth', 'table']

# Mapeos para variables categóricas
CUT_MAPPING = {
    'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4
}

COLOR_MAPPING = {
    'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6
}

CLARITY_MAPPING = {
    'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7
}

@app.route('/')
def home():
    return render_template('index.html')

def preparar_datos_para_modelo(form_data):
    """
    Prepara los datos en el orden exacto que esperan los modelos
    """
    # Extraer y validar datos del formulario
    carat = float(form_data['carat'])
    depth = float(form_data['depth'])
    table = float(form_data['table'])
    x = float(form_data['x'])
    y = float(form_data['y'])
    z = float(form_data['z'])
    cut = form_data['cut']
    color = form_data['color']
    clarity = form_data['clarity']
    
    # Validar categorías
    if cut not in CUT_MAPPING:
        raise ValueError(f"Tipo de corte no válido: {cut}")
    if color not in COLOR_MAPPING:
        raise ValueError(f"Color no válido: {color}")
    if clarity not in CLARITY_MAPPING:
        raise ValueError(f"Claridad no válida: {clarity}")
    
    # Calcular valores derivados
    volume = x * y * z
    clarity_encoded = CLARITY_MAPPING[clarity]
    color_encoded = COLOR_MAPPING[color]
    
    # Crear datos en el ORDEN EXACTO que espera el modelo
    # ['volume', 'clarity_encoded', 'carat', 'color_encoded', 'depth', 'table']
    data_row = [
        volume,           # posición 0
        clarity_encoded,  # posición 1
        carat,           # posición 2
        color_encoded,   # posición 3
        depth,           # posición 4
        table            # posición 5
    ]
    
    return pd.DataFrame([data_row], columns=REQUIRED_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que los modelos estén cargados
    if modelo_rf is None or modelo_mlp is None:
        return jsonify({'error': 'Modelos no disponibles'}), 500
    
    try:
        # Validar campos requeridos
        required_fields = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
        for field in required_fields:
            if field not in request.form or not request.form[field]:
                return jsonify({'error': f'Campo requerido faltante: {field}'}), 400
        
        # Validar rangos básicos
        carat = float(request.form['carat'])
        if not (0.1 <= carat <= 5.0):
            return jsonify({'error': 'Quilates debe estar entre 0.1 y 5.0'}), 400
        
        depth = float(request.form['depth'])
        if not (40 <= depth <= 80):
            return jsonify({'error': 'Profundidad debe estar entre 40 y 80'}), 400
        
        table = float(request.form['table'])
        if not (40 <= table <= 80):
            return jsonify({'error': 'Mesa debe estar entre 40 y 80'}), 400
        
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        
        if x <= 0 or y <= 0 or z <= 0:
            return jsonify({'error': 'Las dimensiones deben ser mayores a 0'}), 400
        
        modelo_usado = request.form.get('modelo', 'rf')
        
        # Preparar datos en el formato correcto
        data_df = preparar_datos_para_modelo(request.form)
        
        app.logger.debug(f"Datos preparados para el modelo:")
        app.logger.debug(f"Columnas: {list(data_df.columns)}")
        app.logger.debug(f"Valores: {data_df.iloc[0].values}")
        
        # Realizar predicción
        if modelo_usado == 'mlp':
            prediction = modelo_mlp.predict(data_df)
            modelo_nombre = 'Red Neuronal'
        else:
            prediction = modelo_rf.predict(data_df)
            modelo_nombre = 'Random Forest'
        
        precio_estimado = float(prediction[0])
        
        app.logger.debug(f"Predicción exitosa: ${precio_estimado:.2f} usando {modelo_nombre}")
        
        return jsonify({
            'precio_estimado': precio_estimado,
            'modelo_usado': modelo_nombre,
            'caracteristicas_usadas': REQUIRED_FEATURES,
            'datos_enviados': data_df.iloc[0].to_dict()
        })
        
    except ValueError as e:
        app.logger.error(f"Error de validación: {str(e)}")
        return jsonify({'error': f'Datos inválidos: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/debug')
def debug_info():
    """
    Endpoint para ver información de debug
    """
    info = {
        'modelos_cargados': {
            'random_forest': modelo_rf is not None,
            'red_neuronal': modelo_mlp is not None
        },
        'caracteristicas_requeridas': REQUIRED_FEATURES,
        'orden_caracteristicas': 'volume, clarity_encoded, carat, color_encoded, depth, table'
    }
    
    if modelo_rf:
        info['random_forest_info'] = {
            'tipo': str(type(modelo_rf)),
            'n_features': getattr(modelo_rf, 'n_features_in_', 'No disponible'),
            'feature_names': list(getattr(modelo_rf, 'feature_names_in_', []))
        }
    
    if modelo_mlp:
        info['red_neuronal_info'] = {
            'tipo': str(type(modelo_mlp)),
            'n_features': getattr(modelo_mlp, 'n_features_in_', 'No disponible'),
            'feature_names': list(getattr(modelo_mlp, 'feature_names_in_', []))
        }
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/')
def home():
    return render_template('index.html')

def preparar_datos_para_modelo(form_data, features):
    """
    Prepara los datos según las características que espera el modelo
    """
    # Extraer datos del formulario
    carat = float(form_data['carat'])
    depth = float(form_data['depth'])
    table = float(form_data['table'])
    x = float(form_data['x'])
    y = float(form_data['y'])
    z = float(form_data['z'])
    cut = form_data['cut']
    color = form_data['color']
    clarity = form_data['clarity']
    
    # Preparar datos según las características requeridas
    data_row = []
    for feature in features:
        if feature == 'carat':
            data_row.append(carat)
        elif feature == 'depth':
            data_row.append(depth)
        elif feature == 'table':
            data_row.append(table)
        elif feature == 'x':
            data_row.append(x)
        elif feature == 'y':
            data_row.append(y)
        elif feature == 'z':
            data_row.append(z)
        elif feature == 'cut':
            data_row.append(cut)
        elif feature == 'color':
            data_row.append(color)
        elif feature == 'clarity':
            data_row.append(clarity)
        elif feature == 'cut_encoded':
            data_row.append(CUT_MAPPING[cut])
        elif feature == 'color_encoded':
            data_row.append(COLOR_MAPPING[color])
        elif feature == 'clarity_encoded':
            data_row.append(CLARITY_MAPPING[clarity])
        elif feature == 'volume':
            data_row.append(x * y * z)
        else:
            raise ValueError(f"Característica desconocida: {feature}")
    
    return pd.DataFrame([data_row], columns=features)

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que los modelos estén cargados
    if modelo_rf is None or modelo_mlp is None:
        return jsonify({'error': 'Modelos no disponibles'}), 500
    
    # Verificar que se hayan detectado las características
    if WORKING_FEATURES_RF is None or WORKING_FEATURES_MLP is None:
        return jsonify({'error': 'No se pudieron detectar las características del modelo'}), 500
    
    try:
        # Validar datos de entrada
        required_fields = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
        for field in required_fields:
            if field not in request.form or not request.form[field]:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        # Validar rangos básicos
        carat = float(request.form['carat'])
        if not (0.1 <= carat <= 5.0):
            return jsonify({'error': 'Quilates debe estar entre 0.1 y 5.0'}), 400
        
        depth = float(request.form['depth'])
        if not (40 <= depth <= 80):
            return jsonify({'error': 'Profundidad debe estar entre 40 y 80'}), 400
        
        table = float(request.form['table'])
        if not (40 <= table <= 80):
            return jsonify({'error': 'Mesa debe estar entre 40 y 80'}), 400
        
        # Validar categorías
        cut = request.form['cut']
        if cut not in CUT_MAPPING:
            return jsonify({'error': f'Tipo de corte no válido: {cut}'}), 400
        
        color = request.form['color']
        if color not in COLOR_MAPPING:
            return jsonify({'error': f'Color no válido: {color}'}), 400
        
        clarity = request.form['clarity']
        if clarity not in CLARITY_MAPPING:
            return jsonify({'error': f'Claridad no válida: {clarity}'}), 400
        
        modelo_usado = request.form.get('modelo', 'rf')
        
        # Preparar datos según el modelo seleccionado
        if modelo_usado == 'mlp':
            if WORKING_FEATURES_MLP is None:
                return jsonify({'error': 'Red Neuronal no disponible'}), 500
            data_df = preparar_datos_para_modelo(request.form, WORKING_FEATURES_MLP)
            prediction = modelo_mlp.predict(data_df)
            modelo_nombre = 'Red Neuronal'
            caracteristicas_usadas = WORKING_FEATURES_MLP
        else:
            if WORKING_FEATURES_RF is None:
                return jsonify({'error': 'Random Forest no disponible'}), 500
            data_df = preparar_datos_para_modelo(request.form, WORKING_FEATURES_RF)
            prediction = modelo_rf.predict(data_df)
            modelo_nombre = 'Random Forest'
            caracteristicas_usadas = WORKING_FEATURES_RF
        
        precio_estimado = float(prediction[0])
        
        app.logger.debug(f"Predicción exitosa: ${precio_estimado:.2f} usando {modelo_nombre}")
        app.logger.debug(f"Características usadas: {caracteristicas_usadas}")
        app.logger.debug(f"Datos enviados: {data_df.iloc[0].to_dict()}")
        
        return jsonify({
            'precio_estimado': precio_estimado,
            'modelo_usado': modelo_nombre,
            'caracteristicas_usadas': caracteristicas_usadas
        })
        
    except ValueError as e:
        app.logger.error(f"Error de validación: {str(e)}")
        return jsonify({'error': f'Datos inválidos: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/debug')
def debug_info():
    """
    Endpoint para ver información de debug
    """
    info = {
        'modelos_cargados': {
            'random_forest': modelo_rf is not None,
            'red_neuronal': modelo_mlp is not None
        },
        'caracteristicas_detectadas': {
            'random_forest': WORKING_FEATURES_RF,
            'red_neuronal': WORKING_FEATURES_MLP
        }
    }
    
    if modelo_rf:
        info['random_forest_info'] = {
            'tipo': str(type(modelo_rf)),
            'n_features': getattr(modelo_rf, 'n_features_in_', 'No disponible'),
            'feature_names': getattr(modelo_rf, 'feature_names_in_', 'No disponible')
        }
    
    if modelo_mlp:
        info['red_neuronal_info'] = {
            'tipo': str(type(modelo_mlp)),
            'n_features': getattr(modelo_mlp, 'n_features_in_', 'No disponible'),
            'feature_names': getattr(modelo_mlp, 'feature_names_in_', 'No disponible')
        }
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)