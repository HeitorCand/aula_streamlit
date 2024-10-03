# Predição de Gasto do Cliente com Machine Learning

Este projeto utiliza modelos de aprendizado de máquina, como **Random Forest** e **Árvore de Decisão**, para realizar predições de gasto de clientes com base em dados fornecidos em um arquivo CSV. A aplicação foi construída utilizando **Streamlit** para criar uma interface interativa, permitindo o upload de arquivos CSV e a seleção de features e colunas alvo para o treinamento do modelo.

## Funcionalidades
- Upload de arquivos CSV com dados de clientes.
- Seleção de colunas para features e coluna alvo (gasto do cliente).
- Treinamento de modelos de **Random Forest** e **Árvore de Decisão**.
- Visualização dos resultados das previsões com gráficos interativos.
- Suporte para gráficos ordenados por data, caso os dados contenham uma coluna de data.

## Requisitos

- **Python 3.7+**
- **Bibliotecas Python** listadas no `requirements.txt`.

## Como rodar o projeto

Siga os passos abaixo para configurar e rodar o projeto localmente:

### 1. Clonar o repositório

Clone este repositório em sua máquina local:

```bash
git clone https://github.com/HeitorCand/aula_streamlit
```

### 2. Navegar até o diretório do projeto

```bash
cd aula_streamlit
```

### 3. Instalar as dependências

3.1 Ambiente virtual (opcional, mas recomendado)
Crie e ative um ambiente virtual utilizando venv:

windows:
```bash
python -m venv venv
source venv/bin/activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3.2 Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```


### 4. Rodar a aplicação

Execute o comando abaixo para rodar a aplicação:

```bash
streamlit run main.py
```


