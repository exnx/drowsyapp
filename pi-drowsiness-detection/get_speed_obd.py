import obd
# from obd import OBDStatus
import time

# or pass which usb port to use
# connection = obd.Async()

# auto connect
connection = obd.Async(baudrate=115200)
print(connection.status() == obd.OBDStatus.CAR_CONNECTED)

# a callback that prints every new value to the console
def new_speed(r):
    print(time.time())
    print(r.value) # returns unit-bearing values thanks to Pint
    print(float(str(r.value.to("mph")))[:2])) # user-friendly unit conversions

# keep track of the Speed
connection.watch(obd.commands.SPEED, callback=new_speed)

# start the async update loop
connection.start()

# the callback will now be fired upon receipt of new values
time.sleep(60)

# stop the async update loop
connection.stop()
