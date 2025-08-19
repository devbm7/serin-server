# Performance Timing and Monitoring

## Overview

The conversational AI pipeline now includes comprehensive timing and performance monitoring features that track how long each process takes. This helps identify bottlenecks, optimize performance, and monitor system behavior.

## Timing Features

### 1. **Detailed Process Timing**
- **ASR Processing**: Tracks preprocessing, generation, and decoding times
- **LLM Response**: Monitors request sending and response processing
- **TTS Synthesis**: Measures audio generation and concatenation
- **Audio Playback**: Tracks actual playback duration
- **Model Loading**: Times each model initialization

### 2. **Performance Summaries**
- Individual utterance processing breakdowns
- Session-wide performance statistics
- Average times per operation
- Total processing times

### 3. **Logging Integration**
All timing information is logged to `pipeline_debug.log` with detailed breakdowns.

## Timing Breakdowns

### ASR Processing
```
ASR preprocessing completed in 0.045s
ASR generation completed in 1.234s
ASR decoding completed in 0.012s
ASR transcription completed in 1.291s: 'Hello, how are you?'
```

### LLM Processing
```
Ollama request sent in 0.023s
Ollama response completed in 2.456s: length=156
```

### TTS Processing
```
TTS generation completed in 0.789s
TTS audio concatenation completed in 0.034s
TTS synthesis completed in 0.823s: audio_length=18944
```

### Complete Utterance Processing
```
=== UTTERANCE PROCESSING SUMMARY ===
ASR: 1.291s
LLM: 2.456s
TTS: 0.823s
Playback: 0.789s
TOTAL: 5.359s
=====================================
```

## Performance Monitoring

### Model Loading Summary
```
=== MODEL LOADING SUMMARY ===
ASR: 15.234s
LLM Selection: 0.123s
TTS: 2.456s
VAD: 1.789s
TOTAL: 19.602s
=============================
```

### Session Summary
```
=== CONTINUOUS PROCESSING SUMMARY ===
Total processing time: 45.678s
Chunks processed: 1250
Utterances processed: 8
Average time per utterance: 5.710s
=====================================
```

## Performance Analysis

### Expected Performance Ranges

#### ASR Models (Whisper)
- **whisper-tiny**: 0.5-1.5s
- **whisper-base**: 1.0-2.5s
- **whisper-small**: 1.5-3.0s
- **whisper-medium**: 3.0-6.0s
- **whisper-large**: 5.0-10.0s

#### LLM Models (Ollama)
- **Small models (0.6B-1B)**: 1.0-3.0s
- **Medium models (1.5B-3B)**: 2.0-5.0s
- **Large models (7B+)**: 3.0-8.0s

#### TTS Processing
- **Text-to-speech**: 0.5-2.0s (depending on text length)
- **Audio playback**: Real-time (matches audio duration)

### Performance Optimization Tips

1. **For Speed**: Use smaller models
   - ASR: `whisper-tiny` or `whisper-base`
   - LLM: `qwen3:0.6b` or `gemma3:1b`

2. **For Quality**: Use larger models
   - ASR: `whisper-large-v3`
   - LLM: `gemma3:latest` or `deepseek-r1:1.5b`

3. **Hardware Considerations**
   - GPU acceleration significantly improves ASR performance
   - More RAM allows larger LLM models
   - SSD storage improves model loading times

## Log Analysis

### Key Performance Indicators

1. **Total Response Time**: Should be under 10 seconds for good UX
2. **ASR Accuracy**: Check transcription quality vs. speed trade-offs
3. **LLM Response Quality**: Balance between speed and coherence
4. **TTS Naturalness**: Audio quality vs. generation speed

### Common Performance Issues

1. **Slow ASR**: 
   - Check GPU availability
   - Consider smaller Whisper model
   - Verify audio input quality

2. **Slow LLM**:
   - Check Ollama server status
   - Consider smaller model
   - Monitor system resources

3. **Slow TTS**:
   - Check text length
   - Verify TTS pipeline configuration
   - Monitor audio output

## Utility Functions

### PerformanceTimer Class
```python
from timing_utils import PerformanceTimer

timer = PerformanceTimer("My Operation")
timer.start()
# ... perform operation ...
duration = timer.stop()
```

### Timed Operations
```python
from timing_utils import timed_operation

with timed_operation("Data Processing"):
    # ... process data ...
```

### Performance Monitor
```python
from timing_utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("operation1")
# ... perform operation ...
monitor.stop_timer("operation1")
monitor.print_summary()
```

## Log File Analysis

The timing logs are written to `pipeline_debug.log`. You can analyze them using:

```bash
# View timing summaries
grep "SUMMARY" pipeline_debug.log

# View ASR timing
grep "ASR" pipeline_debug.log

# View LLM timing
grep "Ollama" pipeline_debug.log

# View TTS timing
grep "TTS" pipeline_debug.log
```

## Performance Benchmarks

### Baseline Performance (Standard Hardware)
- **Model Loading**: 15-25 seconds
- **ASR Processing**: 1-3 seconds per utterance
- **LLM Response**: 2-5 seconds per response
- **TTS Synthesis**: 0.5-2 seconds per response
- **Total Response Time**: 4-10 seconds

### Optimized Performance (High-end Hardware)
- **Model Loading**: 5-15 seconds
- **ASR Processing**: 0.5-1.5 seconds per utterance
- **LLM Response**: 1-3 seconds per response
- **TTS Synthesis**: 0.3-1 second per response
- **Total Response Time**: 2-6 seconds

## Troubleshooting

### Performance Issues
1. Check log files for timing breakdowns
2. Identify slowest component
3. Consider model size reduction
4. Verify hardware resources
5. Check for background processes

### Log Analysis
1. Look for timing summaries
2. Compare with baseline performance
3. Identify bottlenecks
4. Monitor trends over time
5. Set performance alerts

## Future Enhancements

- Real-time performance monitoring dashboard
- Automatic performance optimization suggestions
- Performance regression detection
- Resource usage monitoring
- Performance benchmarking tools 