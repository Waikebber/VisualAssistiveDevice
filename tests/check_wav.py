import wave

with wave.open('redbone.wav', 'rb') as wav_file:
    print(f"Sample width: {wav_file.getsampwidth() * 8} bits")
    print(f"Compression type: {wav_file.getcomptype()}")

# For 32-bit float, sample width will be 4 bytes (32 bits)
# For 16-bit, sample width will be 2 bytes (16 bits)