extends Node

## Emits the latest normalized MMA7660 vector to any interested gameplay script.
signal vector_updated(vector: Vector3)
signal connection_changed(connected: bool)

const DEVICE_NAME: String = "WandBLE"
const SERVICE_UUID: String = "19b10000-e8f2-537e-4f6c-d104768a1214"
const VECTOR_UUID: String = "19b10001-e8f2-537e-4f6c-d104768a1214"

var sensor_vector: Vector3 = Vector3.ZERO
var wand_connected: bool = false

var _manager
var _device
var _found_device_address: String = ""


func _ready() -> void:
	# BluetoothManager and BleDevice are provided by the GDBLE addon.
	_manager = BluetoothManager.new()
	add_child(_manager)

	_manager.adapter_initialized.connect(_on_adapter_initialized)
	_manager.device_discovered.connect(_on_device_discovered)
	_manager.ble_event.connect(_on_manager_ble_event)

	_manager.initialize()


func _on_adapter_initialized(success: bool, error: String) -> void:
	if not success:
		push_error("BLE initialization failed: " + error)
		return

	print("BLE initialized. Scanning for ", DEVICE_NAME, "...")
	# A timeout of 0.0 means scan until stop_scan() is called in GDBLE.
	_manager.start_scan(0.0)


func _on_device_discovered(info: Dictionary) -> void:
	if str(info.get("name", "")) != DEVICE_NAME:
		return

	var address: String = str(info.get("address", ""))
	if address.is_empty() or address == _found_device_address:
		return

	_found_device_address = address
	_manager.stop_scan()
	print("Found wand at ", address, "; connecting...")

	_device = _manager.get_or_create_device(address)
	_device.connected.connect(_on_device_connected)
	_device.ble_event.connect(_on_device_ble_event)
	_device.connect_async()


func _on_device_connected() -> void:
	_set_connection_state(true)
	print("Connected to wand; discovering BLE services...")
	_device.discover_services()


func _on_device_ble_event(event: Dictionary) -> void:
	var operation: String = str(event.get("operation", ""))
	var phase: String = str(event.get("phase", ""))

	if phase == "failed":
		var error_info: Variant = event.get("error", {})
		push_warning("Wand BLE operation failed (%s): %s" % [operation, str(error_info)])
		if operation == "connect" or operation == "disconnect":
			_set_connection_state(false)
		return

	if operation == "discover_services" and phase == "succeeded":
		print("Services discovered; subscribing to vector notifications...")
		_device.subscribe_characteristic(SERVICE_UUID, VECTOR_UUID)
		return

	if operation == "subscribe" and phase == "succeeded":
		print("Subscribed to wand vector notifications.")
		return

	if operation == "disconnect" and (phase == "succeeded" or phase == "received"):
		_set_connection_state(false)
		return

	if operation == "notification" and phase == "received":
		var characteristic_uuid: String = str(
			event.get("characteristic_uuid", "")
		).to_lower()
		if characteristic_uuid != VECTOR_UUID:
			return

		var data: Variant = event.get("data")
		if data is PackedByteArray:
			_decode_vector(data)
		else:
			push_warning("Received BLE notification with unexpected data type.")


func _on_manager_ble_event(event: Dictionary) -> void:
	# Keep manager-level failures visible during setup/debugging.
	if str(event.get("phase", "")) == "failed":
		var operation: String = str(event.get("operation", ""))
		var error_info: Variant = event.get("error", {})
		push_warning("BLE manager event failed (%s): %s" % [operation, str(error_info)])


func _decode_vector(data: PackedByteArray) -> void:
	if data.size() != 6:
		push_warning("Expected 6-byte vector packet; received %d bytes." % data.size())
		return

	var x: float = float(_read_int16_le(data, 0)) / 1000.0
	var y: float = float(_read_int16_le(data, 2)) / 1000.0
	var z: float = float(_read_int16_le(data, 4)) / 1000.0

	sensor_vector = Vector3(x, y, z)
	if sensor_vector.length_squared() > 0.001:
		sensor_vector = sensor_vector.normalized()

	vector_updated.emit(sensor_vector)


func _read_int16_le(data: PackedByteArray, offset: int) -> int:
	var value: int = int(data[offset]) | (int(data[offset + 1]) << 8)
	if value >= 32768:
		value -= 65536
	return value


func _set_connection_state(connected: bool) -> void:
	if wand_connected == connected:
		return
	wand_connected = connected
	connection_changed.emit(connected)


func reconnect_scan() -> void:
	# GDBLE does not automatically reconnect. Call this after a disconnect
	# if you want to restart discovery without restarting the whole game.
	_found_device_address = ""
	_device = null
	_manager.start_scan(0.0)
