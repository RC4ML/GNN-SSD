from os import popen, system
import getpass

ssds = []

if not getpass.getuser() == "root":
    print("\033[41mERROR: Please run this script as root!\033[0m")
    quit(1)

f = popen("lspci | grep Non-Volatile")
lines = f.readlines()
for line in lines:
    ssds.append(line.split(" ")[0])

for idx, ssd in enumerate(ssds):
    # if idx == 0:
    #     continue
    # if idx == 1 or idx == 8 or idx == 6 or idx == 7:
        # continue
    # if idx == 3 or idx == 6 or idx == 9 or idx == 12:
        # continue
    # if idx == 13:
        # continue
    # 4-3 
    # if idx == 1 or idx == 2 or idx == 3 or idx == 8:
        # continue
    if idx == 12:
        continue
    tmp = ssd.replace(":", "\\:")
    system(f"""sh -c 'echo -n "0000:{ssd}" > /sys/bus/pci/devices/0000\:{tmp}/driver/unbind'""")

