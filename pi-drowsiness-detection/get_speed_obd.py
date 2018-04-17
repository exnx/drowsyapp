import obd
import time

# auto connect
connection = obd.Async()

# or pass which usb port to use
connection = obd.Async("/dev/ttyUSB0")

# a callback that prints every new value to the console
def new_speed(r):
    print(r.value) # returns unit-bearing values thanks to Pint
    print(r.value.to("mph")) # user-friendly unit conversions

# keep track of the Speed
connection.watch(obd.commands.SPEED, callback=new_speed)

# start the async update loop
connection.start()

# the callback will now be fired upon receipt of new values


time.sleep(60)

# stop the async update loop
connection.stop()