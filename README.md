# Build-A-Board-Game

This project uses a regression model to score board games by rating. Then, with local search algorithms, it generates a board game predicted to achieve a high rating based on the model’s insights.

## Instructions to Run the Jupyter Notebook

1. **Download the Dataset**  
   Download `board_games.csv` and place it in the home drive of your Google Drive.

2. **Run the Notebook on Google Colab**  
   - Download the `.ipynb` file and upload it to Google Colab within your Google Drive.
   - In Google Colab, select **"Run all"** to execute the notebook.

## Instructions to Run the Python File

1. **Download the Files**  
   Download `board_games.csv` and `predictiveModel.py` and place them in the same folder.

2. **Install Required Libraries**  
   In the terminal, run the following commands to install necessary libraries:
   ```bash
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install prettytable

3. **Run the Script**  
   Execute the Python file in the terminal using:
   ```bash
   python predictiveModel.py
