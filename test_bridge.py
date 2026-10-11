import time
from spellcasting import GodotBridge, SpellRecognizer

bridge = GodotBridge()  # 127.0.0.1:4242 by default
print("Sending to", bridge.addr)

for pose, spell in SpellRecognizer.SPELLS.items():
    print("sending", spell["name"])
    bridge.send_spell(pose, spell)
    time.sleep(1.5)

bridge.close()