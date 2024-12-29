from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt  
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)


DATASET_PATH = '/home/lupita/Documentos/ForestRegresion/TotalFeatures-ISCXFlowMeter.csv'

@app.route('/')
def index():
    
    df = pd.read_csv(DATASET_PATH)
    
 
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

   
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    
    num_training_samples = min(5000, len(X_train))  # Evitar exceder el tamaño del dataset
    X_train = X_train[:num_training_samples]
    y_train = y_train[:num_training_samples]

   
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

  
    try:
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, model.predict(X_test), color='blue')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Resultados de Predicción')
        plot_path = os.path.join('static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Error al generar el gráfico de predicciones: {e}")
        return f"Error al generar el gráfico de predicciones: {e}"

   
    tree_svg_path = 'static/arbol.svg'

    # Verificar que el archivo SVG existe
    if not os.path.exists(tree_svg_path):
        print(f"El archivo SVG no se encuentra en la ruta esperada: {tree_svg_path}")
        return f"Error: El archivo SVG no se encuentra en la ruta {tree_svg_path}"

    # Renderizar la plantilla con el gráfico de predicciones y el archivo SVG
    return render_template('index.html', plot_url=plot_path, tree_svg_url=tree_svg_path)

if __name__ == '__main__':
    app.run(debug=True)
