from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
with open("model_xgb_opt.sav", "rb") as f:
    model = pickle.load(f)

# Definir las columnas requeridas para la predicción
required_columns = ['direccionviento', 'velocidadviento', 'humrelativa', 'tempmaxima', 'diasultimalluvia']

# Definir un umbral (por ejemplo, 0.5)
THRESHOLD = 10

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Recoger datos del formulario y convertirlos a float
            input_data_dict = {}
            for col in required_columns:
                input_data_dict[col] = [float(request.form[col])]
            
            # Crear un DataFrame con los datos de entrada (un solo registro)
            input_data = pd.DataFrame(input_data_dict)
            
            # Realizar la predicción (valor continuo)
            prediction = model.predict(input_data)[0]
            result = prediction
        except Exception as e:
            result = f"Error en los datos: {e}"
    
    # Pasar el resultado y el umbral a la plantilla
    return render_template("index.html", result=result, threshold=THRESHOLD)

if __name__ == "__main__":
    app.run(debug=True)
