# AI-Powered CSV Enrichment Tool

The **AI-Powered CSV Enrichment Tool** is a Streamlit-based application designed to help you analyze, enrich, and visualize your CSV data using advanced AI techniques. With this tool, you can:

- **Data Overview:** View a preview of your CSV file, obtain basic insights, and analyze missing values.
- **AI Analysis:** Leverage Google Generative AI (Gemini) to analyze CSV columns and suggest potential insights.
- **Enrichment Suggestions:** Receive recommendations for enriching your dataset based on sample data.
- **Visualizations:** Create interactive visualizations (scatter plots, bar charts, line charts, and box plots) using Plotly.
- **Chat with CSV:** Ask questions about your data and get concise, insightful answers generated by AI.

## Features

- **CSV Upload:** Easily upload your CSV file for analysis.
- **Automated Analysis:** Generate basic statistics and missing value counts.
- **AI-Driven Insights:** Use generative AI to analyze column data and offer enrichment suggestions.
- **Interactive Visualizations:** Create various plots to explore your dataset.
- **Conversational Interface:** Interact with your data through a chat interface to ask specific questions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/ai-csv-enrichment-tool.git
   cd ai-csv-enrichment-tool
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Create a `.env` file in the project root.
   - Add your Google API Key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```
   - Alternatively, you can add the API key to your Streamlit secrets.

## Usage

1. **Run the Application:**

   ```bash
   streamlit run main.py
   ```

2. **Navigate the Tool:**
   - Use the sidebar navigation to switch between:
     - **Data Overview:** Inspect the CSV file and view basic insights.
     - **AI Analysis:** Generate an AI-driven analysis of the CSV columns.
     - **Enrichment Suggestions:** Get recommendations to enrich your dataset.
     - **Visualizations:** Create interactive charts and plots.
     - **Chat with CSV:** Ask questions about your data and get AI-generated responses.

## Project Structure

```
ai-csv-enrichment-tool/
│
├── main.py                 # Main Streamlit application
├── .env                   # Environment variables file (create this file)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web interface.
- **Google Generative AI:** To generate insights and enrichment suggestions.
- **Pandas:** For CSV data manipulation.
- **Plotly:** For interactive visualizations.
- **Python-Dotenv:** For environment variable management.
- **Streamlit Option Menu:** For intuitive sidebar navigation.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests with improvements or bug fixes.
