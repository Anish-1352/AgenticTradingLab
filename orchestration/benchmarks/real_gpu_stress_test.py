import concurrent.futures
import time
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" # Upgraded to 7B model since we have an A100
MAX_WORKERS = 15 # Concurrent threads for 105 tasks
TASKS = 105      # Total tasks to process

print(f"[Setup] Loading {MODEL_ID} into GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Simulating thousands of tokens
THICK_CONTEXT_BLOCK = "--- ANNUAL EARNINGS REPORT & FORWARD GUIDANCE ---\n" * 200 

def run_inference(task_id):
    start_time = time.time()
    
    prompt = f"{THICK_CONTEXT_BLOCK}\n\nTask {task_id}: Analyze the above report and output 'BUY' or 'SELL'."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids.shape[1]
    
    try:
        # Forward pass & generation
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True)
        output_tokens = outputs.shape[1] - input_tokens
        
        latency = time.time() - start_time
        
        return {
            "task_id": task_id,
            "status": "Success",
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    except RuntimeError as e:
        # Catch CUDA Out Of Memory Errors
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {"task_id": task_id, "status": "OOM", "latency": time.time() - start_time, "error": "CUDA OOM"}
        return {"task_id": task_id, "status": "Error", "latency": time.time() - start_time, "error": str(e)}

def run_profiler():
    print("\n🔍 Running 2-step torch.profiler slice...")
    prompt = "Task profiler: Analyze and output."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/content/tb_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(2):
            model.generate(**inputs, max_new_tokens=5, do_sample=False)
            prof.step()
    print("✅ Profiler trace saved to /content/tb_logs/")

def main():
    print(f"\n🚀 Starting GPU Stress Test: {TASKS} tasks across {MAX_WORKERS} concurrent threads...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"Initial System RAM Used: {psutil.virtual_memory().used / (1024**2):.2f} MB\n")

    start_time = time.time()
    results = []
    total_tokens_processed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_inference, i) for i in range(TASKS)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            if res["status"] == "Success":
                total_tokens_processed += (res["input_tokens"] + res["output_tokens"])
                print(f"Task {res['task_id']:02d} | Latency: {res['latency']:>5.2f}s | Tokens: {res['input_tokens']}")
            else:
                print(f"Task {res['task_id']:02d} | ❌ FAILED: {res['error']}")
                
    total_time = time.time() - start_time
    
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    print("\n" + "="*50)
    print("📊 TRUE HARDWARE METRICS SUMMARY")
    print("="*50)
    print(f"Total Execution Time:  {total_time:.2f} seconds")
    print(f"Total Tokens Handled:  {total_tokens_processed:,}")
    if total_time > 0:
        print(f"True Token Throughput: {total_tokens_processed / total_time:,.2f} tokens/sec")
    
    success_latencies = [r['latency'] for r in results if r['status'] == 'Success']
    if success_latencies:
        print(f"Latency Variance:      Min {min(success_latencies):.2f}s  |  Max {max(success_latencies):.2f}s")
    print(f"Peak VRAM Allocated:   {peak_vram:,.2f} MB")
    print("="*50)
    
    # Run profiler step
    run_profiler()

if __name__ == '__main__':
    main()
