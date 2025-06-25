#!/usr/bin/env python3
import joblib
import pandas as pd

def test_exact_features():
    """
    Prueba con las características exactas detectadas
    """
    print("🧪 Probando con las características exactas...")
    
    try:
        rf = joblib.load('modelo_random_forest.pkl')
        mlp = joblib.load('modelo_red_neuronal.pkl')
        print("✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return
    
    # Mapeos
    CLARITY_MAPPING = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    COLOR_MAPPING = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    
    # Datos de ejemplo
    carat = 0.50
    depth = 61.5
    table = 55.0
    x, y, z = 4.05, 4.07, 2.31
    cut, color, clarity = 'Ideal', 'E', 'SI1'
    
    # Calcular valores
    volume = x * y * z
    clarity_encoded = CLARITY_MAPPING[clarity]
    color_encoded = COLOR_MAPPING[color]
    
    # ORDEN EXACTO según el debug: ['volume', 'clarity_encoded', 'carat', 'color_encoded', 'depth', 'table']
    features = ['volume', 'clarity_encoded', 'carat', 'color_encoded', 'depth', 'table']
    values = [volume, clarity_encoded, carat, color_encoded, depth, table]
    
    print(f"\n📊 Datos de prueba:")
    for feature, value in zip(features, values):
        print(f"   {feature}: {value}")
    
    try:
        df = pd.DataFrame([values], columns=features)
        print(f"\n✅ DataFrame creado: {df.shape}")
        
        # Probar Random Forest
        pred_rf = rf.predict(df)[0]
        print(f"✅ Random Forest: ${pred_rf:.2f}")
        
        # Probar Red Neuronal
        pred_mlp = mlp.predict(df)[0]
        print(f"✅ Red Neuronal: ${pred_mlp:.2f}")
        
        print(f"\n🎉 ¡PERFECTO! Todo funciona correctamente")
        print(f"📋 Las características correctas son: {features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_exact_features()