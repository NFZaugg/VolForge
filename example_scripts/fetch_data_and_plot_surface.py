from datetime import date

from matplotlib import pyplot as plt

from vol_forge import VolForgeClient

if __name__ == "__main__":
    base_date = date(2026, 7, 24)
    expiry_date = date(2026, 8, 21)
    max_date = date(2026, 9, 30)
    symbol = "AAPL"

    client = VolForgeClient()

    print(client.fetch_ivs_for_date(symbol, base_date, expiry_date))

    fig, ax = client.plot_single_expiry(symbol, base_date, expiry_date)
    plt.show()

    fig, ax = client.plot_surface(symbol, base_date, expiry_date, max_date)
    plt.show()
