# Open Source Stock Price Checker and Query

This Python script creates an open-source stock price checker and query system using web scraping, natural language processing, and vector embeddings.

## Features

- Web scraping using Jina AI's service
- Web search functionality using Tavily Search API
- Document processing and chunking
- Vector embeddings and storage using Chroma
- Natural language querying using Groq's language model

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.x installed
- Required Python packages (install using `pip install -r requirements.txt`)
- API keys for Groq, HuggingFace, and Tavily

## Setup

1. Clone the repository:

git clone https://github.com/yourusername/stock-price-checker.git
cd stock-price-checker

2. Install the required packages:

pip install -r requirements.txt

3. Create a `.env` file in the project root and add your API keys:

GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
TAVILY_API_KEY=your_tavily_api_key

(Keep all the api's in " ". Eg : "sk-9321abde")

## Usage

1. Run the script:

Preferably, use jupyter notebook.

2. In the **FOURTH** cell, change the question as per your liking.


3. The system will then:

- Perform a web search using Tavily Search API
- Scrape relevant web pages using Jina AI's service
- Process and chunk the scraped text
- Create vector embeddings and store them in Chroma
- Use Groq's language model to generate an answer based on the processed information

4. The answer to your query will be displayed in the console.

## Limitations

- The script is designed to answer questions specifically related to the stock market and stock prices. It will not provide information on unrelated topics.
- The accuracy of the information depends on the web search results and the most recent data available online.
- The script uses the current date and time for queries that require temporal context.

## Contributing

Contributions to improve the script are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Disclaimer

This script is for educational and informational purposes only. It should not be used as a sole source for making financial decisions. Always consult with a qualified financial advisor before making investment choices.

## In case you have an OPENAI API KEY...

You can play around with stonk_openai.py. Simply change the question in **line 17** and run the code using 

python stonk_openai.py