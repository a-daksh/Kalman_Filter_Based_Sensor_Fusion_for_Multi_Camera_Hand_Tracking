import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cam_mixed.csv')
 
serial_number = data['Serial No.']
sensor1_data = data['Camera1']
sensor2_data = data['Camera2']
fused_sensor_data = data['Fused']
# average=data['Average']

plt.plot(serial_number, sensor1_data, label='Lenovo')
plt.plot(serial_number, sensor2_data, label='Logitech')
plt.plot(serial_number, fused_sensor_data, label='Fused')
# plt.plot(serial_number, average, label='Average')

plt.xlabel('Time Steps')
plt.ylabel('Sensor Data')
plt.title('Sensor Data vs Time')

plt.legend()

plt.show()
