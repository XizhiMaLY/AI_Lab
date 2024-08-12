import trace

# Create a Trace object
tracer = trace.Trace(
    trace=True,  # Enable tracing
    count=False  # Disable line counting
)

# Run the script with tracing enabled
tracer.run('python /home/agent_mxz/AI_Lab/qwen2_7b/start_head.py')
