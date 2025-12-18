# AskSQL: Anime Database Query System

## Project Overview

This project implements a sophisticated **Natural Language to SQL (Text-to-SQL)** system designed to query an anime database. By fine-tuning a T5-small transformer model, the system translates natural language user questions (e.g., "What are the top 5 rated action anime?") into executable SQL queries.

The system features a **Flask-based web interface** for real-time interaction, a robust **synthetic data generation pipeline**, and efficient **LoRA (Low-Rank Adaptation)** fine-tuning to ensure high performance with minimal computational resources.

## Key Features

*   **Natural Language Interface**: Users can query the database using everyday English.
*   **Advanced Model Fine-tuning**: Uses **T5-small** with **LoRA** (Low-Rank Adaptation) via the `peft` library for efficient parameter-efficient training.
*   **Synthetic Data Generation**: A custom pipeline creates diverse, schema-aware training examples to handle various SQL patterns (filtering, sorting, aggregations).
*   **Interactive Web UI**: A clean, responsive web interface built with **Flask** and **Bootstrap** to visualize queries and results.
*   **Schema Awareness**: Implementation includes schema linking strategies to improve column name prediction accuracy.

## Dataset

The project uses a comprehensive dataset of anime information.
*   **Source Data**: [Kaggle Anime Rating Dataset](https://www.kaggle.com/datasets/vishnujg/anime-rating)
*   **Local File**: The project uses a processed version named `data/animenew.csv` which is automatically loaded into a local SQLite database (`project_db.sqlite`) upon initialization.

## Technical Architecture

*   **Model**: Google's T5-small (Text-to-Text Transfer Transformer).
*   **Fine-tuning**: PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
*   **Backend**: Python Flask Server.
*   **Database**: SQLite.
*   **Frontend**: HTML/CSS/JavaScript (Jinja2 Templates).

## Installation

1.  **Clone the Repository** (if applicable)

2.  **Create Custom Environment**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Mac/Linux
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies: `torch`, `transformers`, `peft`, `flask`, `pandas`.*

## Usage

### 1. Initialize Database & Prepare Data
The system requires `data/animenew.csv`. To initialize the SQLite database and generate synthetic training data:

```bash
# Initialize DB and generate training pairs
python scripts/generate_synthetic.py
```
This script acts as the data engine, producing `data/train_processed.csv` and `data/dev_processed.csv`.

### 2. Train the Model (Fine-Tuning)
Train the T5 model using the generated data. This script applies LoRA configuration and saves the best model to `models/`.

```bash
python scripts/train_model.py
```
*Note: Training parameters (epochs, batch size) can be adjusted in the script.*

### 3. Run the Web Application
Start the Flask server to interact with the system.

```bash
python server.py
```
*   Open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)
*   Enter questions like:
    *   *"List top 5 rated anime"*
    *   *"How many anime have usage > 1000?"*
    *   *"Show me action movies with rating above 8"*

## Project Structure

```text
├── data/
│   ├── animenew.csv            # Raw source data
│   ├── train_processed.csv     # Generated training data
│   └── dev_processed.csv       # Generated development data
├── models/                     # Saved model artifacts (after training)
├── scripts/
│   ├── generate_synthetic.py   # Data generation pipeline
│   ├── train_model.py          # T5 + LoRA training script
│   ├── preprocess.py           # Utility for data formatting
│   └── test_project.py         # Unit tests
├── src/
│   ├── database.py             # DB connection & initialization logic
│   └── inference.py            # Model loading & SQL generation logic
├── templates/
│   └── index.html              # Web interface template
├── server.py                   # Main Flask application entry point
├── project_db.sqlite           # Local SQLite database (auto-generated)
└── requirements.txt            # Python dependencies
```

## Testing

To verify the system components (inference engine and database connection) are working correctly:

```bash
python scripts/test_project.py
```
