extends Node

## Attach this Node as a child of the player, not as an Autoload.
## Expected scene tree:
## Player (CharacterBody3D or Node3D)
## ├── Head (Node3D)
## │   └── Camera3D
## └── WandCameraController (Node)

@export var player_body_path: NodePath = NodePath("..")
@export var camera_path: NodePath = NodePath("../Head/Camera3D")

@export_group("Camera response")
@export var max_tilt_degrees: float = 25.0
@export var yaw_speed: float = 120.0
@export var pitch_speed: float = 90.0
@export_range(0.0, 0.5, 0.01) var deadzone: float = 0.10
@export var smoothing: float = 10.0
@export var invert_yaw: bool = false
@export var invert_pitch: bool = false
@export var min_pitch_degrees: float = -80.0
@export var max_pitch_degrees: float = 80.0

@onready var player_body: Node3D = get_node_or_null(player_body_path) as Node3D
@onready var camera: Camera3D = get_node_or_null(camera_path) as Camera3D

var latest_vector: Vector3 = Vector3.ZERO
var smoothed_vector: Vector3 = Vector3.ZERO
var has_vector: bool = false
var calibrated: bool = false
var neutral_roll: float = 0.0
var neutral_pitch: float = 0.0
var last_packet_time_ms: int = 0


func _ready() -> void:
	if player_body == null:
		push_error("WandCameraController: player body not found. Check player_body_path.")
		set_process(false)
		return

	if camera == null:
		push_error("WandCameraController: Camera3D not found. Check camera_path.")
		set_process(false)
		return

	if not WandReceiver.vector_updated.is_connected(_on_wand_vector_updated):
		WandReceiver.vector_updated.connect(_on_wand_vector_updated)

	print("Wand camera controller ready. Hold the wand at neutral on first data.")


func _on_wand_vector_updated(vector: Vector3) -> void:
	if vector.length_squared() < 0.001:
		return

	latest_vector = vector.normalized()
	last_packet_time_ms = Time.get_ticks_msec()

	if not has_vector:
		smoothed_vector = latest_vector
		has_vector = true

	# The first good packet defines the neutral wand pose.
	# Hold the wand comfortably in your desired neutral position at startup.
	if not calibrated:
		var angles: Vector2 = _get_tilt_angles(latest_vector)
		neutral_roll = angles.x
		neutral_pitch = angles.y
		calibrated = true
		print("Wand neutral pose calibrated. Tilt to look around.")


func _process(delta: float) -> void:
	if not has_vector or not calibrated:
		return

	# Freeze control if BLE notifications stop arriving.
	if Time.get_ticks_msec() - last_packet_time_ms > 500:
		return

	var blend: float = 1.0 - exp(-maxf(smoothing, 0.0) * delta)
	smoothed_vector = smoothed_vector.lerp(latest_vector, blend)

	var angles: Vector2 = _get_tilt_angles(smoothed_vector)
	var roll_delta: float = wrapf(angles.x - neutral_roll, -PI, PI)
	var pitch_delta: float = angles.y - neutral_pitch

	var max_tilt: float = deg_to_rad(max_tilt_degrees)
	if max_tilt <= 0.0:
		return

	var yaw_input: float = clampf(roll_delta / max_tilt, -1.0, 1.0)
	var pitch_input: float = clampf(pitch_delta / max_tilt, -1.0, 1.0)

	yaw_input = _apply_deadzone(yaw_input)
	pitch_input = _apply_deadzone(pitch_input)

	var yaw_sign: float = -1.0 if invert_yaw else 1.0
	var pitch_sign: float = -1.0 if invert_pitch else 1.0

	# Tilting left/right steers yaw; holding a tilt keeps turning.
	player_body.rotate_y(
		yaw_input * deg_to_rad(yaw_speed) * yaw_sign * delta
	)

	# Tilting forward/back steers pitch. Camera is clamped to avoid flipping.
	camera.rotation.x += (
		pitch_input * deg_to_rad(pitch_speed) * pitch_sign * delta
	)
	camera.rotation.x = clampf(
		camera.rotation.x,
		deg_to_rad(min_pitch_degrees),
		deg_to_rad(max_pitch_degrees)
	)


func _get_tilt_angles(vector: Vector3) -> Vector2:
	# These are gravity-based tilt estimates, NOT a full orientation estimate.
	# If your MMA7660 is mounted differently, you may need to swap/invert axes.
	var roll: float = atan2(vector.y, vector.z)
	var pitch: float = atan2(
		-vector.x,
		sqrt(vector.y * vector.y + vector.z * vector.z)
	)
	return Vector2(roll, pitch)


func _apply_deadzone(value: float) -> float:
	var amount: float = absf(value)
	if amount < deadzone:
		return 0.0
	return signf(value) * ((amount - deadzone) / (1.0 - deadzone))
