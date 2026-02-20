from level_3.demo_aac_3 import demo_aac_3

SNR, bitrate, compression = demo_aac_3("LicorDeCalandraca.wav", "output_level3.wav", "coded_level3.mat")
print(f"Level 3 \n SNR = {SNR:.2f} dB \n Bitrate = {bitrate/1000:.2f} kbps \n Compression = {compression:.2f}x")
