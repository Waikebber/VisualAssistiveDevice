import smbus

# I2C initialization
bus = smbus.SMBus(1)
imu_address = 0x68  # Default I2C address for the IMU

# Registers for accelerometer
ACCEL_XOUT_H = 0x2D
ACCEL_YOUT_H = 0x2F
ACCEL_ZOUT_H = 0x31

def read_sensor_data():
    """Read raw accelerometer data from the IMU."""
    accel_x = read_axis(ACCEL_XOUT_H)
    accel_y = read_axis(ACCEL_YOUT_H)
    accel_z = read_axis(ACCEL_ZOUT_H)
    return accel_x, accel_y, accel_z

def read_axis(register):
    """Read 16-bit signed value from two consecutive registers."""
    high = bus.read_byte_data(imu_address, register)
    low = bus.read_byte_data(imu_address, register + 1)
    value = (high << 8) | low
    if value >= 0x8000:  # Convert to signed 16-bit
        value = -(65536 - value)
    return value

def apply_calibration(raw_data, offsets, scales):
    """Apply offsets and scaling to raw accelerometer data."""
    calibrated_data = [
        (raw_data[i] - offsets[i]) * scales[i] for i in range(3)
    ]
    return calibrated_data

def load_calibration(filepath):
    """Load calibration data from a JSON file."""
    with open(filepath, "r") as f:
        calibration_data = json.load(f)
    return calibration_data
