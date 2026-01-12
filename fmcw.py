import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.signal import butter, filtfilt    
def cfar_2d(rd_map,
            num_train_r=10, num_guard_r=4,
            num_train_d=6,  num_guard_d=2,
            threshold_scale=5.5):
    """
    2D CA-CFAR on Range-Doppler map (in dB)
    rd_map: 2D numpy array (doppler x range)
    returns: binary detection map
    """
    detections = np.zeros_like(rd_map, dtype=int)

    num_doppler, num_range = rd_map.shape

    for d in range(num_train_d + num_guard_d,
                   num_doppler - (num_train_d + num_guard_d)):
        for r in range(num_train_r + num_guard_r,
                       num_range - (num_train_r + num_guard_r)):

            noise_cells = []

            for dd in range(d - num_train_d - num_guard_d,
                            d + num_train_d + num_guard_d + 1):
                for rr in range(r - num_train_r - num_guard_r,
                                r + num_train_r + num_guard_r + 1):

                    # Skip guard cells and CUT
                    if (abs(dd - d) <= num_guard_d and
                        abs(rr - r) <= num_guard_r):
                        continue

                    noise_cells.append(rd_map[dd, rr])

            noise_level = np.mean(noise_cells)
            threshold = noise_level + threshold_scale

            if rd_map[d, r] > threshold:
                detections[d, r] = 1

    return detections


# FMCW RADAR PARAMETERS (AUTOMOTIVE-LIKE)

c = 3e8                 # Speed of light (m/s)
fc = 77e9               # Carrier frequency (Hz)
B = 200e6               # Bandwidth (Hz)
T = 40e-6               # Chirp duration (s)
S = B / T               # Chirp slope (Hz/s)
fs = 20e6               # Sampling frequency (Hz)

R = 50                  # Target range (m)
v = 10                  # Target velocity (m/s)
num_chirps = 64         # Chirps per frame

print("FMCW radar simulation started")

# =====================================================
# TIME AXIS
# =====================================================
t = np.arange(0, T, 1/fs)

# =====================================================
# TRANSMIT CHIRP (BASEBAND)
# =====================================================
tx = np.cos(2 * np.pi * (S / 2) * t**2)

# =====================================================
# DAY 2 — BEAT SIGNAL (SINGLE CHIRP)
# =====================================================
tau = 2 * R / c
rx = np.cos(2 * np.pi * (S / 2) * (t - tau)**2)
beat = tx * rx

plt.figure()
plt.plot(t * 1e6, beat)
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("Beat Signal")
plt.tight_layout()
plt.savefig("01_beat_signal.png", dpi=300)
plt.close()

# =====================================================
# DAY 3 — RANGE FFT
# =====================================================
beat_dc = beat - np.mean(beat)
beat_win = beat_dc * windows.hann(len(beat_dc))

N_fft = 8192
fft_vals = np.fft.fft(beat_win, N_fft)
fft_mag = np.abs(fft_vals)

freqs = np.fft.fftfreq(N_fft, d=1/fs)
half = N_fft // 2

ranges = (c * freqs[:half]) / (2 * S)

plt.figure()
plt.plot(ranges, fft_mag[:half])
plt.xlim(0, 120)
plt.xlabel("Distance (m)")
plt.ylabel("Amplitude")
plt.title("Range FFT")
plt.grid()
plt.tight_layout()
plt.savefig("02_range_fft.png", dpi=300)
plt.close()

# =====================================================
# DAY 4 — RANGE–DOPPLER MAP (WITH REAL DOPPLER PHYSICS)
# =====================================================
beat_matrix = np.zeros((num_chirps, len(t)))

# Doppler frequency (THIS WAS MISSING BEFORE)
f_D = 2 * v * fc / c
b, a = butter(4, 0.25) #4th order Butterworth filter
for k in range(num_chirps):
    R_k = R + v * k * T
    tau_k = 2 * R_k / c

    rx_k = np.cos(
        2 * np.pi * (S / 2) * (t - tau_k)**2
        + 2 * np.pi * f_D * (k * T + t)   # ✅ Doppler phase
    )

    beat_matrix[k, :] = tx * rx_k
    beat_matrix[k, :] = filtfilt(b, a, beat_matrix[k, :] )  # Apply filter

# Windowing in range dimension
beat_matrix *= windows.hann(len(t))

# Range FFT (keep complex)
range_fft = np.fft.fft(beat_matrix, axis=1)
# Doppler windowing
doppler_window = np.hanning(num_chirps)
range_fft *= doppler_window[:, None]

# Doppler FFT (slow time)
doppler_fft = np.fft.fftshift(
    np.fft.fft(range_fft, axis=0),
    axes=0
)
doppler_fft = np.abs(doppler_fft)

# =====================================================
# AXES
# =====================================================
freq_r = np.fft.fftfreq(len(t), d=1/fs)
ranges = (c * freq_r) / (2 * S)
ranges_pos = ranges[:len(ranges)//2]

doppler_freq = np.fft.fftshift(
    np.fft.fftfreq(num_chirps, d=T)
)
velocity_axis = doppler_freq * (c / (2 * fc))

doppler_fft_pos = doppler_fft[:, :len(ranges)//2]
# =========================
# DAY 5 — CFAR DETECTION
# =========================

# Convert Range–Doppler map to dB scale
rd_map_db = 20 * np.log10(doppler_fft_pos + 1e-6)

# Apply 2D CA-CFAR
detections = cfar_2d(
    rd_map_db,
    num_train_r=10,
    num_guard_r=4,
    num_train_d=6,
    num_guard_d=2,
    threshold_scale=5.5 # Adjusted for dB scale
)
print("CFAR detection completed")
print("doppler_fft_pos.shape")
# =========================
# PEAK PICKING (FINAL STEP)
# =========================

# Get indices of CFAR detections
det_doppler_idx, det_range_idx = np.where(detections == 1)

if len(det_range_idx) == 0:
    print("No target detected by CFAR.")
else:
    # Extract power values at detected cells
    detected_powers = rd_map_db[det_doppler_idx, det_range_idx]

    # Find strongest detection
    strongest_idx = np.argmax(detected_powers)

    best_d = det_doppler_idx[strongest_idx]
    best_r = det_range_idx[strongest_idx]

    # Convert indices to physical values
    estimated_range = ranges_pos[best_r]
    estimated_velocity = velocity_axis[best_d]

    print("===== FINAL TARGET ESTIMATE =====")
    print(f"Range    : {estimated_range:.2f} m")
    print(f"Velocity : {estimated_velocity:.2f} m/s")

# =====================================================
# SAVE RANGE–DOPPLER MAP
# =====================================================
plt.figure(figsize=(9,6))
plt.imshow(
    doppler_fft_pos,
    aspect="auto",
    extent=[
        0, np.max(ranges_pos),
        velocity_axis[0], velocity_axis[-1]
    ],
    origin="lower",
    cmap="jet"
)
plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range–Doppler Map")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.savefig("03_range_doppler.png", dpi=300)
plt.close()

print("Simulation complete")
print("Generated files:")
print(" - 01_beat_signal.png")
print(" - 02_range_fft.png")
print(" - 03_range_doppler.png")
print(" - 04_cfar_detections.png")

# CFAR DETECTION
detections = cfar_2d(doppler_fft_pos)

plt.figure(figsize=(9,6))
plt.imshow(
    doppler_fft_pos,
    aspect ="auto",
    extent=[0,np.max(ranges_pos),velocity_axis[0],velocity_axis[-1]],
    origin="lower", 
    cmap="jet"

)
det_y,det_x = np.where(detections==1)
plt.scatter(ranges_pos[det_x],
   velocity_axis[det_y],
 c ="red" ,s =15, label ="CFAR Detections" 
 )  
plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range-Doppler Map with CFAR Detections")
plt.colorbar(label="Amplitude")
plt.legend()    
plt.tight_layout()
plt.savefig("04_cfar_detections.png", dpi=300)
plt.close()