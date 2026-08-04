import time
import torch
import os
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LOG_DIR = "/content/tb_logs_advanced"
os.makedirs(LOG_DIR, exist_ok=True)

print(f"[Setup] Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

THICK_CONTEXT = "--- ANNUAL EARNINGS REPORT ---\n" * 100 + "\nAnalyze this data and give a 10 word summary."
inputs = tokenizer(THICK_CONTEXT, return_tensors="pt").to(model.device)

def profile_generation(task_id):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=15, do_sample=False)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    start_time = time.time()
    thread.start()
    
    first_token_time = None
    token_count = 0
    
    for new_text in streamer:
        if first_token_time is None and new_text.strip():
            first_token_time = time.time()
        if new_text.strip():
            token_count += 1
            
    end_time = time.time()
    thread.join()
    
    ttft = first_token_time - start_time if first_token_time else 0
    tpot = (end_time - first_token_time) / token_count if token_count > 0 else 0
    
    print(f"Task {task_id} | TTFT: {ttft*1000:.2f} ms | TPOT: {tpot*1000:.2f} ms/token | Tokens: {token_count}")

print("\n🚀 Running Advanced Metrics test...")
# Run a few warmup iterations
for i in range(2):
    profile_generation(f"Warmup-{i}")

print("\n🔍 Starting torch.profiler...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i in range(4): # 1 wait, 1 warmup, 2 active
        profile_generation(f"Trace-{i}")
        prof.step()

print(f"\n✅ Profiler trace saved successfully to {LOG_DIR}/")
