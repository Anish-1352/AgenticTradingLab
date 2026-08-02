import concurrent.futures
import time
import random
from run_simple_backtest import generate_market_data, run_trading_strategy, calculate_metrics, save_results
from datetime import datetime

THICK_CONTEXT_BLOCK = """
--- ANNUAL EARNINGS REPORT & FORWARD GUIDANCE ---
""" * 500 # Simulating a large context block (approx thousands of tokens)

def simulate_single_run(agent, symbol, run_idx):
    print(f"[Worker {run_idx}] Starting {agent} on {symbol} with thick context...\n", end="")
    
    # Simulate passing thick context block to agent logic (mocked here as a sleep penalty)
    time.sleep(random.uniform(0.5, 2.0))
    
    market_data = generate_market_data(symbol, days=252)
    equity_curve, trades, daily_returns, drawdowns = run_trading_strategy(market_data, initial_equity=100000)
    metrics = calculate_metrics(equity_curve, trades, daily_returns, drawdowns, initial_equity=100000)
    
    print(f"[Worker {run_idx}] ✅ {agent} on {symbol} completed. Return: {metrics['total_return']*100:+.2f}%\n", end="")
    return {
        'agent': agent,
        'symbol': symbol,
        'return': f"{metrics['total_return']*100:.2f}%",
        'sharpe': f"{metrics['sharpe_ratio']:.2f}"
    }

def main():
    agents = ['DeepSeek', 'Claude', 'GPT-4']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    # Create artificially multiplied load
    tasks = [(agent, symbol, i) for i, agent in enumerate(agents * 5) for symbol in symbols]
    
    print(f"🚀 Starting Concurrent Stress Test with {len(tasks)} total tasks across 15 threads...")
    
    start_time = time.time()
    results_summary = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_task = {executor.submit(simulate_single_run, t[0], t[1], t[2]): t for t in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results_summary.append(result)
            except Exception as exc:
                print(f"Task generated an exception: {exc}")
                
    print(f"\n⏱️ All tasks completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
