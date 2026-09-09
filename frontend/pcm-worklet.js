// Capture mono PCM at the AudioContext's 16 kHz rate. Flush the last partial
// frame before finish so final words are not silently truncated.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer = new Int16Array(4000); this.offset = 0; this.active = false;
    this.port.onmessage = ({data}) => {
      if (data === 'start') { this.offset = 0; this.active = true; }
      if (data === 'finish') {
        if (this.offset) this.emit();
        this.active = false; this.port.postMessage({finished: true});
      }
      if (data === 'cancel') { this.offset = 0; this.active = false; }
    };
  }
  emit() {
    const pcm = this.buffer.slice(0, this.offset);
    let energy = 0;
    for (const sample of pcm) energy += (sample / 32768) ** 2;
    this.port.postMessage({pcm: pcm.buffer, rms: Math.sqrt(energy / Math.max(1, pcm.length))}, [pcm.buffer]);
    this.offset = 0;
  }
  process(inputs) {
    if (this.active && inputs[0]?.[0]) {
      for (const sample of inputs[0][0]) {
        this.buffer[this.offset++] = Math.round(Math.max(-1, Math.min(1, sample)) * 32767);
        if (this.offset === this.buffer.length) this.emit();
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
