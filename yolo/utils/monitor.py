import psutil
import time
import logging
import torch
import pynvml

def monitor_resources(throttle_temp=80, pause_duration=300):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    temp = get_temperature()

    logging.info(f'[MONITOR] CPU: {cpu}%, Memory: {mem}%, Temp: {temp}°C')

    if temp and temp > throttle_temp:
        logging.warning(f'[OVERHEAT] Temp {temp}°C > {throttle_temp}°C. Pausing...')
        time.sleep(pause_duration)

def get_temperature():
    try:
        temps = psutil.sensors_temperatures()
        for name in temps:
            if 'core' in name.lower():
                return temps[name][0].current
        return None
    except Exception:
        return None