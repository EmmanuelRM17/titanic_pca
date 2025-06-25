from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar ambos modelos entrenados
modelo_rf = joblib.load('modelo_random_forest.pkl')
modelo_mlp = joblib.load('modelo_red_neuronal.pkl')
app.logger.debug("Modelos cargados correctamente.")

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener atributos del formulario (ajusta si necesitas más o menos)
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        volume = float(request.form['volume'])
        cut_encoded = int(request.form['cut_encoded'])
        color_encoded = int(request.form['color_encoded'])
        clarity_encoded = int(request.form['clarity_encoded'])

        # Modelo a usar (opcional)
        modelo_usado = request.form.get('modelo', 'rf')  # default es rf

        # Crear DataFrame con los datos
        data_df = pd.DataFrame([[
            carat, depth, table, volume, cut_encoded, color_encoded, clarity_encoded
        ]], columns=[
            'carat', 'depth', 'table', 'volume', 'cut_encoded', 'color_encoded', 'clarity_encoded'
        ])
        app.logger.debug(f"Datos recibidos: {data_df}")

        # Seleccionar modelo
        if modelo_usado == 'mlp':
            prediction = modelo_mlp.predict(data_df)
        else:
            prediction = modelo_rf.predict(data_df)

        app.logger.debug(f"Predicción: {prediction[0]}")

        return jsonify({'precio_estimado': float(prediction[0])})
    
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
