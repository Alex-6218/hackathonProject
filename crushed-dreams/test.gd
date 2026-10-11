extends Node

func _ready() -> void:
	print(
		"GDBLE loaded: ",
		ClassDB.class_exists("BluetoothManager")
	)
