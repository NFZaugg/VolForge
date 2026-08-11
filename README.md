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

Examples

<img width="2000" height="1200" alt="slice" src="https://github.com/user-attachments/assets/beec6465-8d20-49e5-a4dd-987843fcf5cd" />


<img width="2000" height="1200" alt="surface" src="https://github.com/user-attachments/assets/bc4befa6-aabd-4c62-bf0d-d8453b3a1814" />
