import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, butter, filtfilt

# =====================================================
# CFAR FUNCTION
# =====================================================
def cfar_2d(rd_map,
            num_train_r=12, num_guard_r=6,
            num_train_d=8,  num_guard_d=4,
            threshold_scale=6.0):

    detections = np.zeros_like(rd_map, dtype=int)
    D, R = rd_map.shape

    for d in range(num_train_d + num_guard_d, D - (num_train_d + num_guard_d)):
        for r in range(num_train_r + num_guard_r, R - (num_train_r + num_guard_r)):

            noise_cells = []

            for dd in range(d - num_train_d - num_guard_d, d + num_train_d + num_guard_d + 1):
                for rr in range(r - num_train_r - num_guard_r, r + num_train_r + num_guard_r + 1):
                    if abs(dd - d) <= num_guard_d and abs(rr - r) <= num_guard_r:
                        continue
                    noise_cells.append(rd_map[dd, rr])

            threshold = np.mean(noise_cells) + threshold_scale
            if rd_map[d, r] > threshold:
                detections[d, r] = 1

    return detections


# =====================================================
# RADAR PARAMETERS
# =====================================================
c = 3e8
fc = 77e9
B = 200e6
T = 40e-6
S = B / T
fs = 20e6

R = 50.0
v = -10.0
num_chirps = 64
SNR_dB = 15

print("FMCW radar simulation started")

# =====================================================
# TIME AXIS
# =====================================================
t = np.arange(0, T, 1/fs)

# =====================================================
# TRANSMIT CHIRP (IQ BASEBAND)
# =====================================================
tx = np.exp(1j * 2 * np.pi * (S / 2) * t**2)

# =====================================================
# DEMO: SINGLE-CHIRP BEAT SIGNAL (FOR VISUALIZATION)
# =====================================================
tau_demo = 2 * R / c
rx_demo = np.exp(1j * 2 * np.pi * (S / 2) * (t - tau_demo)**2)
beat_demo = tx * np.conj(rx_demo)

plt.figure()
plt.plot(t * 1e6, np.real(beat_demo))
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("Beat Signal (Single Chirp)")
plt.grid()
plt.tight_layout()
plt.savefig("01_beat_signal.png", dpi=300)
plt.close()

# =====================================================
# RANGE–DOPPLER SIGNAL GENERATION
# =====================================================
beat_matrix = np.zeros((num_chirps, len(t)), dtype=complex)

f_D = 2 * v * fc / c
b, a = butter(4, 0.25)

for k in range(num_chirps):

    R_k = R + v * k * T
    tau_k = 2 * R_k / c

    rx_k = np.exp(
        1j * (
            2 * np.pi * (S / 2) * (t - tau_k)**2
            + 2 * np.pi * f_D * (k * T + t)
        )
    )

    beat = tx * np.conj(rx_k)
    attenuation = 1 / (R_k**2)
    beat *= attenuation
   

    signal_power = np.mean(np.abs(beat)**2)
    noise_power = signal_power / (10**(SNR_dB / 10))

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )

    beat = beat + noise
    beat = filtfilt(b, a, beat)

    beat_matrix[k, :] = beat

# =====================================================
# RANGE FFT
# =====================================================
beat_matrix *= windows.hann(len(t))[None, :]

range_fft = np.fft.fft(beat_matrix, axis=1)
range_fft = range_fft[:, :len(t)//2]

freq_r = np.fft.fftfreq(len(t), d=1/fs)[:len(t)//2]
ranges = (c * freq_r) / (2 * S)

# ---- RANGE FFT PLOT (FROM FIRST CHIRP) ----
plt.figure()
plt.plot(ranges, np.abs(range_fft[0, :]))
plt.xlim(0, 120)
plt.xlabel("Distance (m)")
plt.ylabel("Amplitude")
plt.title("Range FFT (Single Chirp)")
plt.grid()
plt.tight_layout()
plt.savefig("02_range_fft.png", dpi=300)
plt.close()

# =====================================================
# DOPPLER FFT
# =====================================================
range_fft *= windows.hann(num_chirps)[:, None]

doppler_fft = np.fft.fftshift(
    np.fft.fft(range_fft, axis=0),
    axes=0
)

rd_map = np.abs(doppler_fft)
max_range_m = 120
valid_range_bins = ranges <= max_range_m
# extract valid range bins
rd_map_valid = rd_map[:, valid_range_bins]
ranges = ranges[valid_range_bins]
# normalize
#range_mean = np.mean(rd_map_valid, axis=0, keepdims=True)
#range_mean = np.maximum(range_mean, 1e-6)
#rd_map_norm = rd_map_valid / range_mean

# Convert to dB FIRST
rd_map_db = 20 * np.log10(rd_map_valid + 1e-6)

# Remove slow range trend (high-pass in range)
range_trend = np.mean(rd_map_db, axis=0, keepdims=True)
rd_map_db_detrended = rd_map_db - range_trend

# CFAR detection
detections = cfar_2d(rd_map_db, num_train_r=12, num_guard_r=6,
                     num_train_d=8, num_guard_d=4,
                     threshold_scale=6.0)
snr_gate_db = 6
detections  &= (rd_map_db_detrended >= snr_gate_db)
#compute velocity axis
doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_chirps, d=T))
velocity_axis = doppler_freq * (c / (2 * fc))

# =====================================================
# PEAK PICKING
# =====================================================
det_d, det_r = np.where(detections == 1)

if len(det_r) > 0:
    powers = rd_map_db[det_d, det_r]
    idx = np.argmax(powers)

    est_range = ranges[det_r[idx]]
    est_velocity = velocity_axis[det_d[idx]]

    print("===== FINAL TARGET ESTIMATE =====")
    print(f"Range    : {est_range:.2f} m")
    print(f"Velocity : {est_velocity:.2f} m/s")
else:
    print("No target detected")

# =====================================================
# RANGE–DOPPLER MAP PLOT
# =====================================================
plt.figure(figsize=(9,6))
plt.imshow(
    rd_map_db,
    aspect="auto",
    extent=[0, np.max(ranges),
            velocity_axis[0], velocity_axis[-1]],
    origin="lower",
    cmap="jet"
)
plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range–Doppler Map")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()
plt.savefig("03_range_doppler.png", dpi=300)
plt.close()

# =====================================================
# RANGE–DOPPLER MAP WITH CFAR
# =====================================================
plt.figure(figsize=(9,6))
plt.imshow(
    rd_map_db,
    aspect="auto",
    extent=[0, np.max(ranges),
            velocity_axis[0], velocity_axis[-1]],
    origin="lower",
    cmap="jet"
)

plt.scatter(
    ranges[det_r],
    velocity_axis[det_d],
    c="red", s=15, label="CFAR Detections"
)

plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range–Doppler Map with CFAR")
plt.colorbar(label="Amplitude (dB)")
plt.legend()
plt.tight_layout()
plt.savefig("04_cfar_detections.png", dpi=300)
plt.close()

print("Simulation complete")
print("Generated plots:")
print(" - 01_beat_signal.png")
print(" - 02_range_fft.png")
print(" - 03_range_doppler.png")
print(" - 04_cfar_detections.png")
