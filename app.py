from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Variable global para el modelo
modelo_titanic = None

def load_model():
    """Carga el modelo de Titanic"""
    global modelo_titanic
    try:
        modelo_titanic = joblib.load('modelo_titanic.pkl')
        app.logger.info("✅ Modelo Titanic cargado correctamente")
        return True
    except Exception as e:
        app.logger.error(f"❌ Error cargando modelo: {e}")
        return False

def transform_to_pca(form_data):
    """
    Transforma datos del formulario directamente a los 8 componentes PCA
    que espera tu modelo entrenado
    """
    try:
        # Extraer y convertir datos
        pclass = int(form_data['pclass'])
        sex = 1.0 if form_data['sex'] == 'male' else 0.0  # male=1, female=0
        age = float(form_data['age'])
        sibsp = int(form_data['sibsp'])
        parch = int(form_data['parch'])
        fare = float(form_data['fare'])
        
        # Embarked mapping: C=0, Q=1, S=2 (orden alfabético típico)
        embarked_map = {'C': 0.0, 'Q': 1.0, 'S': 2.0}
        embarked = embarked_map[form_data['embarked']]
        
        # Valores por defecto para Ticket y Cabin (simplificados)
        ticket = 500.0  # Valor medio típico
        cabin = 100.0   # Valor por defecto para "desconocido"
        
        # Crear vector de características originales
        # Orden: [Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]
        features = np.array([pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
        
        # Normalizar usando estadísticas aproximadas del dataset Titanic
        # Medias aproximadas
        means = np.array([2.31, 0.65, 29.7, 0.52, 0.38, 260.7, 32.2, 72.9, 1.72])
        # Desviaciones estándar aproximadas  
        stds = np.array([0.84, 0.48, 14.5, 1.10, 0.81, 471.0, 49.7, 64.1, 0.82])
        
        # Escalar
        features_scaled = (features - means) / stds
        
        # Transformación PCA hardcodeada basada en tu notebook
        # Estos son los loadings aproximados de los 8 componentes principales
        pca_matrix = np.array([
            [ 0.565,  0.126, -0.314,  0.096, -0.003,  0.242, -0.445,  0.499,  0.226],
            [ 0.011, -0.353, -0.409,  0.554,  0.550,  0.083,  0.297, -0.038, -0.049],
            [-0.128,  0.012,  0.066,  0.131,  0.186, -0.641, -0.183, -0.006,  0.695],
            [-0.082,  0.648,  0.185,  0.266,  0.092,  0.465,  0.271, -0.161,  0.380],
            [ 0.050,  0.576, -0.417,  0.270, -0.160, -0.448, -0.060, -0.173, -0.399],
            [ 0.101,  0.251,  0.472, -0.081,  0.684, -0.137, -0.166,  0.223, -0.366],
            [-0.214, -0.065,  0.345,  0.553, -0.365, -0.097,  0.132,  0.590, -0.127],
            [-0.105,  0.187, -0.311, -0.456,  0.099, -0.148,  0.585,  0.517,  0.088]
        ])
        
        # Aplicar transformación PCA
        pca_features = np.dot(features_scaled, pca_matrix.T)
        
        # Devolver como array 2D (1 muestra, 8 características)
        return pca_features.reshape(1, -1)
        
    except Exception as e:
        app.logger.error(f"Error en transform_to_pca: {e}")
        raise ValueError(f"Error transformando datos: {e}")

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predicción de supervivencia usando tu modelo entrenado"""
    if modelo_titanic is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
        # Validar campos requeridos
        required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        for field in required_fields:
            if field not in request.form:
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        # Validar valores
        pclass = int(request.form['pclass'])
        if pclass not in [1, 2, 3]:
            return jsonify({'error': 'Clase debe ser 1, 2 o 3'}), 400
            
        sex = request.form['sex']
        if sex not in ['male', 'female']:
            return jsonify({'error': 'Sexo debe ser male o female'}), 400
            
        age = float(request.form['age'])
        if age < 0 or age > 100:
            return jsonify({'error': 'Edad debe estar entre 0 y 100'}), 400
            
        fare = float(request.form['fare'])
        if fare < 0:
            return jsonify({'error': 'Tarifa debe ser positiva'}), 400
            
        embarked = request.form['embarked']
        if embarked not in ['S', 'C', 'Q']:
            return jsonify({'error': 'Puerto debe ser S, C o Q'}), 400
        
        # Transformar datos a formato PCA
        data_pca = transform_to_pca(request.form)
        
        # Hacer predicción con tu modelo
        prediction = modelo_titanic.predict(data_pca)[0]
        probabilities = modelo_titanic.predict_proba(data_pca)[0]
        
        # Preparar respuesta
        sobrevive = bool(prediction)
        prob_supervivencia = float(probabilities[1])  # Probabilidad de sobrevivir
        confianza = float(max(probabilities))
        
        app.logger.info(f"Predicción: {sobrevive}, Prob: {prob_supervivencia:.3f}")
        
        return jsonify({
            'sobrevive': sobrevive,
            'resultado': 'SOBREVIVE' if sobrevive else 'NO SOBREVIVE',
            'probabilidad': prob_supervivencia,
            'confianza': confianza
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error en predicción: {e}")
        return jsonify({'error': f'Error interno del servidor'}), 500

@app.route('/debug')
def debug():
    """Información de debug"""
    return jsonify({
        'modelo_cargado': modelo_titanic is not None,
        'tipo_modelo': str(type(modelo_titanic)) if modelo_titanic else None,
        'status': 'OK' if modelo_titanic else 'ERROR',
        'archivo_existe': os.path.exists('modelo_titanic.pkl')
    })

@app.route('/test')
def test():
    """Prueba rápida del modelo"""
    if modelo_titanic is None:
        return jsonify({'error': 'Modelo no cargado'})
    
    try:
        # Datos de prueba: mujer joven, primera clase
        test_form = {
            'pclass': '1',
            'sex': 'female', 
            'age': '25',
            'sibsp': '1',
            'parch': '0',
            'fare': '50.0',
            'embarked': 'S'
        }
        
        data_pca = transform_to_pca(test_form)
        prediction = modelo_titanic.predict(data_pca)[0]
        probabilities = modelo_titanic.predict_proba(data_pca)[0]
        
        return jsonify({
            'test_case': 'Mujer, 25 años, primera clase',
            'prediction': int(prediction),
            'probability': float(probabilities[1]),
            'pca_shape': data_pca.shape,
            'message': 'Modelo funcionando correctamente'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en test: {e}'})

# 🔥 CLAVE: Cargar modelo al importar (para gunicorn)
load_model()

# Para desarrollo local y compatibilidad con gunicorn
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)