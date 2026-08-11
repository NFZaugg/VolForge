# Welcome to VolForge - An implied volatility surface blacksmith

# Quickstart

- Checkout repo
- Add '.env. file with [Thetadata](https://www.thetadata.net/) API key
  
  THETADATA_API_KEY="your_api_key_here"
  
- run `example_scripts/fetch_data_and_plot_surface.py`
# Features:

- Downloads option price data from Thetadata. (has all OPRA-listed options EOD prices for free)
- Cleans the quotes
- Computes Implied Forward curve
- Constructs implied vol surface for bid/ask/mid - using both Put and Call quotes
- Visualizes the data


<img width="1280" height="960" alt="slice" src="https://github.com/user-attachments/assets/725bc546-c930-42a5-aee1-5a32d7032145" />

