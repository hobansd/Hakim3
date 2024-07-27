import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import poisson, binom
from sklearn.metrics import accuracy_score
import math
from multiprocessing import Pool, cpu_count
import time
from collections import Counter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from prophet import Prophet

# Caminho para o arquivo .xlsx
file_path = r'C:\Users\Hakim\Downloads\mega_sena_asloterias_ate_concurso_2753_sorteio.xlsx'

# Carregar dados históricos da Mega-Sena de um arquivo Excel
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Arquivo carregado com sucesso!")
        return data
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        raise
    except PermissionError:
        print(f"Sem permissão para acessar o arquivo: {file_path}")
        raise
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo: {e}")
        raise

# Pré-processar dados para redes neurais
def preprocess_data(data):
    X = data[['Bola 1', 'Bola 2', 'Bola 3', 'Bola 4', 'Bola 5', 'Bola 6']].values
    y = []
    for row in X:
        binary_row = np.zeros(60)
        for num in row:
            binary_row[num-1] = 1
        y.append(binary_row)
    return X, np.array(y)

# Calcular probabilidades históricas
def calculate_historical_probabilities(X):
    num_counts = np.zeros(60)
    for row in X:
        for num in row:
            num_counts[num-1] += 1
    total_counts = np.sum(num_counts)
    historical_prob = num_counts / total_counts
    return historical_prob

# Definir e treinar a rede neural LSTM adaptativa
def create_and_train_lstm_model(X, y):
    X = np.expand_dims(X, axis=-1)
    model = Sequential([
        LSTM(2048, activation='relu', input_shape=(6, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)  # Ajuste os parâmetros conforme necessário
    return model

# Aplicação do Teorema de Bayes
def bayesian_inference(X, y):
    total_counts = np.sum(y, axis=0)
    total_draws = len(y)
    prior_prob = total_counts / total_draws
    likelihood = np.mean(y, axis=0)
    posterior_prob = (likelihood * prior_prob) / np.sum(likelihood * prior_prob)
    return posterior_prob

# Modelagem usando Distribuições de Poisson e Binomial
def poisson_binomial_modeling(X):
    lambda_poisson = np.mean(X)
    poisson_dist = poisson(lambda_poisson)
    n, p = 60, lambda_poisson / 60
    binom_dist = binom(n, p)
    return poisson_dist, binom_dist

# Implementação de Regressão Logística e Linear Múltipla
def regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    logistic_preds = logistic_model.predict(X_test)
    logistic_acc = accuracy_score(y_test, logistic_preds)
    
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)
    linear_acc = linear_model.score(X_test, y_test)
    
    return logistic_model, linear_model, logistic_acc, linear_acc

# Modelos GARCH
def garch_modeling(X):
    garch_model = arch_model(X, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    return garch_fit

# Modelos SARIMA
def sarima_modeling(X):
    sarima_model = SARIMAX(X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    return sarima_fit

# Modelos Holt-Winters
def holt_winters_modeling(X):
    hw_model = ExponentialSmoothing(X, trend='add', seasonal='add', seasonal_periods=12)
    hw_fit = hw_model.fit()
    return hw_fit

# Modelos Prophet
def prophet_modeling(data):
    df = data[['Data Sorteio', 'Bola 1']].rename(columns={'Data Sorteio': 'ds', 'Bola 1': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return model, forecast

# Simulações de Monte Carlo
def monte_carlo_simulation_single(model, num_simulations):
    simulations = []
    for _ in range(num_simulations):
        random_input = np.random.randint(1, 61, size=(1, 6))
        prediction = model.predict(random_input)
        simulations.append((random_input, prediction))
    return simulations

def monte_carlo_simulation_parallel(model, num_simulations=50063860):
    num_workers = cpu_count()
    simulations_per_worker = num_simulations // num_workers
    with Pool(num_workers) as pool:
        results = pool.starmap(monte_carlo_simulation_single, [(model, simulations_per_worker) for _ in range(num_workers)])
    simulations = [item for sublist in results for item in sublist]
    return simulations

# Analisar simulações de Monte Carlo e encontrar a combinação mais frequente
def find_most_frequent_combination(simulations):
    combination_counter = Counter()
    for _, prediction in simulations:
        predicted_combination = tuple(np.argsort(prediction[0])[-6:] + 1)
        combination_counter[predicted_combination] += 1
    most_frequent_combination = combination_counter.most_common(1)[0]
    return most_frequent_combination

# Combinar probabilidades históricas com resultados das simulações
def combine_probabilities(historical_prob, most_frequent_combination):
    combined_prob = {}
    for combination, freq in most_frequent_combination:
        prob = 1
        for num in combination:
            prob *= historical_prob[num-1]
        combined_prob[combination] = prob * freq
    best_combination = max(combined_prob, key=combined_prob.get)
    return best_combination

# Função principal
def main(file_path):
    start_time = time.time()

    data = load_data(file_path)
    load_data_time = time.time()
    print(f"Tempo para carregar os dados: {load_data_time - start_time} segundos")

    X, y = preprocess_data(data)
    preprocess_time = time.time()
    print(f"Tempo para pré-processar os dados: {preprocess_time - load_data_time} segundos")
    
    lstm_model = create_and_train_lstm_model(X, y)
    lstm_time = time.time()
    print(f"Tempo para treinar o modelo LSTM: {lstm_time - preprocess_time} segundos")

    historical_prob = calculate_historical_probabilities(X)
    historical_prob_time = time.time()
    print(f"Tempo para calcular probabilidades históricas: {historical_prob_time - lstm_time} segundos")

    # Adicionando modelagem GARCH
    garch_fit = garch_modeling(X)
    garch_time = time.time()
    print(f"Tempo para modelar GARCH: {garch_time - historical_prob_time} segundos")

    # Adicionando modelagem SARIMA
    sarima_fit = sarima_modeling(X)
    sarima_time = time.time()
    print(f"Tempo para modelar SARIMA: {sarima_time - garch_time} segundos")

    # Adicionando modelagem Holt-Winters
    hw_fit = holt_winters_modeling(X)
    hw_time = time.time()
    print(f"Tempo para modelar Holt-Winters: {hw_time - sarima_time} segundos")

    # Adicionando modelagem Prophet
    prophet_model, prophet_forecast = prophet_modeling(data)
    prophet_time = time.time()
    print(f"Tempo para modelar Prophet: {prophet_time - hw_time} segundos")

    simulations = monte_carlo_simulation_parallel(lstm_model)
    monte_carlo_time = time.time()
    print(f"Tempo para simulações de Monte Carlo: {monte_carlo_time - prophet_time} segundos")

    most_frequent_combination = find_most_frequent_combination(simulations)
    best_combination = combine_probabilities(historical_prob, [most_frequent_combination])

    print(f"Combinação mais frequente baseada em dados históricos e simulações: {best_combination}")

    # Exemplo de como imprimir alguns resultados
    print(f"Probabilidades históricas: {historical_prob}")
    
    end_time = time.time()
    print(f"Tempo total de execução: {end_time - start_time} segundos")

if __name__ == "__main__":
    main(file_path)
