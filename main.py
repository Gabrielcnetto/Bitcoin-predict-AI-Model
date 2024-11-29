from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

modelo = None

def carregar_modelo(caminho):
    global modelo
    modelo = joblib.load(caminho)
    print("Modelo carregado!")

def prever_preco_btc(dados):
    dados_df = pd.DataFrame([dados], columns=[
        'Open', 'High', 'Low', 'Volume', 'RSI', 'EMA', 'BOL_middle', 'BOL_std',
        'BOL_upper', 'BOL_lower', 'MAVOL1', 'MAVOL2', 'EMA12', 'EMA26', 'MACD', 'MACD_signal',
        'Year', 'Month', 'Day', 'Day_of_week', 'Hour', 'Minute'
    ])
    
    previsao = modelo.predict(dados_df)
    return previsao[0]

@app.route('/previsao', methods=['POST'])
def previsao():
    try:
        dados = request.get_json()
        previsao_btc = prever_preco_btc(dados)
        return jsonify({"previsao": previsao_btc}), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    carregar_modelo('models/RandomForstPredictionBitcoin.pkl')  # Carrega o modelo antes de rodar o servidor
    app.run(host="0.0.0.0", port=5000, debug=True)
